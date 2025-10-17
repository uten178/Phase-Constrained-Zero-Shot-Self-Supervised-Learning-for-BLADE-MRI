#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 22:46:28 2025

@author: ubun
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time, pdb, os
import scipy.io as sio
import glob

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, calculate_density_compensator
dtype = tf.float32

from tensorflow.keras.layers import Lambda

# Deep learning packages
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Activation, Concatenate, BatchNormalization, Add, Subtract, LayerNormalization, \
    LeakyReLU, Conv2D, Convolution2D, MaxPooling2D, UpSampling2D, \
    Conv2DTranspose, Dropout, concatenate, SeparableConv2D, PReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal

#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.2/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
#export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
# ####################################################
def generate_blade_trajactory(N, num_blades):

    Angle = np.linspace(0, np.pi, num_blades,endpoint=False)
        
    x = np.linspace(0, N-1, N)
    y = np.linspace(0, N-1, N)
    
    kx_base,ky_base = np.meshgrid(x,y)
    
    kx_base=-(kx_base/N)+0.5;
    ky_base=-(ky_base/N)+0.5;
    
    k_traj = []
    
    for it in range(num_blades):
        
        angle = Angle[it]
    
        xo=np.cos(angle)*kx_base+np.sin(angle)*ky_base;
        yo=np.cos(angle)*ky_base-np.sin(angle)*kx_base;
    
        kx = xo.flatten()
        ky = yo.flatten()
        ktraj = np.stack((-ky, -kx), axis=0)*2*np.pi
    # convert k-space trajectory to a tensor and unsqueeze batch dimension
        #ktraj = tf.convert_to_tensor(ktraj)[None, ...]
    # create NUFFT objects, use 'ortho' for orthogonal FFTs
        k_traj.append(ktraj)
    
    k_traj = tf.stack(k_traj,axis=-1)
    
    return k_traj
#######################################################################################
data_path = '/home/ubun/Documents/My_CNN_BLADE_LIVER_Project'

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, filename_list,num_rows, num_cols, num_coils, num_blades,batch_size,
                 shuffle=True):

        self.filename_list = filename_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.filename_list)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_coils = num_coils
        self.num_blades = num_blades
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            #Shuffle the filename list in-place
            np.random.shuffle(self.filename_list)

    def __get_data(self, filenames):

        kdata_T = np.empty((self.batch_size,self.num_coils, self.num_rows*self.num_cols,self.num_blades)).astype(np.complex64)
        kdata_L = np.empty((self.batch_size,self.num_coils, self.num_rows*self.num_cols,self.num_blades*2)).astype(np.float32)
        kmask_T = np.empty((self.batch_size,self.num_coils, self.num_rows*self.num_cols,self.num_blades)).astype(np.complex64)
        kmask_L = np.empty((self.batch_size,self.num_coils, self.num_rows*self.num_cols,self.num_blades)).astype(np.complex64)       
        csm = np.empty((self.batch_size,self.num_coils,self.num_rows,self.num_cols)).astype(np.complex64)
        K_coord = np.empty((self.batch_size,2,self.num_rows*self.num_cols,self.num_blades)).astype(np.float32)

        for idx, curr_filename in enumerate(filenames):
            kdata_T[idx,], kdata_L[idx,], kmask_T[idx,], kmask_L[idx,], csm[idx,], K_coord[idx,] = self.prepare_single_input_output_pair(curr_filename)
        return tuple([kdata_T,K_coord,csm,kmask_T]), tuple([kdata_L, K_coord ,csm, kmask_L])

    # Return the index'th batch
    def __getitem__(self, index):
        curr_filenames = self.filename_list[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(curr_filenames)

        return X, y

    def __len__(self):
        return self.num_samples // self.batch_size

    def prepare_single_input_output_pair(self,filename_one_sample):
        temp = sio.loadmat(filename_one_sample)
        #K_coord = temp['K_coord']
        K_coord = generate_blade_trajactory(self.num_rows, 40)## CHECK THIS!!
        K_coord = K_coord[:, :, 0:40:2]## CHECK THIS!!
        kdata_T = temp['kdata_T'].astype(np.csingle)
        kdata_T = kdata_T/np.max(np.abs(kdata_T))
        kdata_L = temp['kdata_L'].astype(np.csingle)
        kdata_L = kdata_L/np.max(np.abs(kdata_L))
        
        maskT = temp['mask_T'].astype(np.csingle)
        maskL = temp['mask_L'].astype(np.csingle)

        csm = temp['csm'].astype(np.csingle)
        #csm = np.transpose(csm,[2,0,1])

        Nc = self.num_coils
        N = self.num_rows
        Np = self.num_blades

        kdata_T = np.transpose(kdata_T,[2,0,1,3])
        kdata_L = np.transpose(kdata_L,[2,0,1,3])
        kmask_T = np.transpose(maskT,[2,0,1,3])
        kmask_L = np.transpose(maskL,[2,0,1,3])

        kdata_T = np.reshape(kdata_T,[Nc,N*N,Np])
        kdata_L = np.reshape(kdata_L,[Nc,N*N,Np])
        kmask_T = np.reshape(kmask_T,[Nc,N*N,Np])
        kmask_L = np.reshape(kmask_L,[Nc,N*N,Np])

        return kdata_T, complex_to_real(kdata_L), kmask_T, kmask_L, csm, K_coord


############################################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Subtract, Lambda, LayerNormalization
from tensorflow.keras.models import Model

########
fft2c = Lambda(lambda x: fft2c_tf(x))
ifft2c = Lambda(lambda x: ifft2c_tf(x))
fft2c_coil = Lambda(lambda x: fft2c_coil_tf(x))
ifft2c_coil = Lambda(lambda x: ifft2c_coil_tf(x))

complex_to_real = Lambda(lambda x: complex_to_real_tf(x))
real_to_complex = Lambda(lambda x: real_to_complex_tf(x))

virtual_coil = Lambda(lambda x: virtual_coil_tf(x))
actual_coil = Lambda(lambda x: actual_coil_tf(x))

#######################################################################################
def virtual_coil_tf(image):
    vc_image = tf.transpose(image,(0,2,1,3),conjugate=True)
    image_out = tf.concat((image,vc_image),axis=3)
    return image_out

def actual_coil_tf(image):
    tmp1 = 0.5*(image[...,0]+tf.transpose(image[...,2],(0,2,1),conjugate=True))
    tmp2 = 0.5*(image[...,1]+tf.transpose(image[...,3],(0,2,1),conjugate=True))
    image_out = tf.stack([tmp1,tmp2],axis = -1)
    return image_out

def complex_to_real_tf(image):
    image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
    shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1]*2]],axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out

def real_to_complex_tf(image):
    image_out = tf.reshape(image, [-1, 2])
    image_out = tf.complex(image_out[:, 0], image_out[:, 1])
    shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] // 2]],axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out


def fft2c_tf(x):
    # x: [batch, row, col] ... x in this case
    # tf.signal.fft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    # Inner-most dimension = right-most dimension

    Fx = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))), axes=(-2, -1))/tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return Fx

def ifft2c_tf(x):
    # x: [batch, row, col] ...k in this case
    # tf.signal.ifft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    Ft_x = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))*tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return Ft_x

# fft2c with coil dimension
def fft2c_coil_tf(x):
    # x: [batch, row, col, coil] ... Cx in this case
    # tf.signal.fft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    # Inner-most dimension = right-most dimension
    # So, we need to swap things around
    x = tf.transpose(x, perm=(0, 3, 1, 2))  # -> [batch, coil, row, col]

    Fx = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))), axes=(-2, -1))/tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Fx, perm=(0, 2, 3, 1))  # -> Back to [batch, row, col, coil]

# ifft2c with coil dimension
def ifft2c_coil_tf(x):
    # x: [batch, row, col, coil] ...Mt_k in this case
    # tf.signal.ifft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    x = tf.transpose(x, perm=(0, 3, 1, 2))  # -> [row,col,coil, batch]
    Ft_x = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))*tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Ft_x, perm=(0, 2, 3, 1))  # -> Back to [batch, row, col, coil]
def A_propeller(img,K_coord,csm,kmask,Np):
  X = []
  for it in range(Np):
      # nufft_ob = Nufft_P[it]
      ktraj = K_coord[:,:,:,it]
      img0 = img[:,:,:,it]
      kmask0 = kmask[:,:,:,it]
      out = compute_A()(img0,ktraj,csm,kmask0)
      X.append(out)

  return tf.stack(X,axis=-1)

def AH_propeller(kdata,K_coord,csm,kmask,Np):
  X = []
  for it in range(Np):
      # nufft_ob = Nufft_P[it]
      ktraj = K_coord[:,:,:,it]
      kdata0 = kdata[:,:,:,it]
      kmask0 = kmask[:,:,:,it]
      out = compute_Ah()(kdata0,ktraj,csm,kmask0)
      X.append(out)
      
  out = tf.stack(X,axis=-1)

  return out

def AHA_propeller(img,K_coord,csm,kmask,Np):
  X = []
  for it in range(Np):
      # nufft_ob = Nufft_P[it]
      ktraj = K_coord[:,:,:,it]
      img0 = img[:,:,:,it]
      kmask0 = kmask[:,:,:,it]
      out = getDataTerm_AhAx()(img0,ktraj,csm,kmask0)
      X.append(out)

  return tf.stack(X,axis=-1)

def transpos_nufft(kdata,ktraj,csm,kmask):
    img = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(tf.multiply(kdata,kmask), ktraj)
    img_sumC = img*tf.math.conj(csm)
    out = tf.reduce_sum(img_sumC,axis=1)

    return out

def forward_nufft(img,ktraj,csm,kmask):
    img = tf.expand_dims(img, axis=1)
    # csm = tf.expand_dims(csm, axis=-1)
    imgC = tf.multiply(img, csm)
    out = kbnufft_forward(nufft_ob._extract_nufft_interpob())(imgC, ktraj)

    return out*kmask
class compute_A(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(compute_A, self).__init__()
        self.forward_nufft = forward_nufft

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(compute_A, self).build(input_shape)

    # call, where you do the forward computation
    def call(self,x,ktraj,csm,kmask):
        # inputs: x, C, M, W

        # Compute A(A(x))
        return self.forward_nufft(x,ktraj,csm,kmask)

class compute_Ah(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(compute_Ah, self).__init__()
        self.transpos_nufft = transpos_nufft

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(compute_Ah, self).build(input_shape)

    # call, where you do the forward computation
    def call(self,x,ktraj,csm,kmask):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return self.transpos_nufft(x,ktraj,csm,kmask)

class getDataTerm_AhAx(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(getDataTerm_AhAx, self).__init__()

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(getDataTerm_AhAx, self).build(input_shape)

    # call, where you do the forward computation
    def call(self,x,ktraj,csm,kmask):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return transpos_nufft(forward_nufft(x,ktraj,csm,kmask),ktraj,csm,kmask)

from tensorflow.keras import layers
###############################################################################

def residual_dense_block(input_tensor, growth_rate, num_layers):
    x = input_tensor
    concat_features = [x]
    for _ in range(num_layers):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    # Local Feature Fusion
    x = layers.Conv2D(growth_rate, (1, 1), padding='same')(x)
    return x

def residual_dense_network(input_shape, out_ch, growth_rate, num_rdb, num_layers_per_rdb,last_filt):# growth_rate=64, num_rdb=5, num_layers_per_rdb=4)
    """
    Creates a Residual Dense Network with ~10M parameters.

    Args:
        input_shape (tuple): Shape of the input tensor (H, W, C).
        growth_rate (int): Number of filters in each dense block.
        num_rdb (int): Number of residual dense blocks.
        num_layers_per_rdb (int): Number of layers in each residual dense block.

    Returns:
        model (tf.keras.Model): Compiled Residual Dense Network model.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(inputs)
    initial_conv = x

    # Residual Dense Blocks
    for _ in range(num_rdb):
        rdb_output = residual_dense_block(x, growth_rate, num_layers_per_rdb)
        x = layers.Add()([x, rdb_output])  # Residual connection

    # Global Feature Fusion
    x = layers.Conv2D(growth_rate, (1, 1), padding='same')(x)
    x = layers.Add()([initial_conv, x])  # Residual connection

    # Final Convolution
    outputs = layers.Conv2D(out_ch, (last_filt, last_filt), padding='same')(x)

    model = Model(inputs, outputs)

    return model
##########################################

#########################################
def UNet(input_shape=(320, 320, 4), out_ch=4, last_filt=3, use_skip=True, use_add=True):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)  # 160x160

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)  # 80x80

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)  # 40x40

    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)  # 20x20

    # Bottleneck
    b = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    b = layers.Conv2D(1024, 3, activation='relu', padding='same')(b)

    # Decoder
    u4 = layers.UpSampling2D(2)(b)  # 40x40
    if use_skip:
        u4 = layers.Concatenate()([u4, c4])
    d4 = layers.Conv2D(512, 3, activation='relu', padding='same')(u4)
    d4 = layers.Conv2D(512, 3, activation='relu', padding='same')(d4)

    u3 = layers.UpSampling2D(2)(d4)  # 80x80
    if use_skip:
        u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv2D(256, 3, activation='relu', padding='same')(u3)
    d3 = layers.Conv2D(256, 3, activation='relu', padding='same')(d3)

    u2 = layers.UpSampling2D(2)(d3)  # 160x160
    if use_skip:
        u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    d2 = layers.Conv2D(128, 3, activation='relu', padding='same')(d2)

    u1 = layers.UpSampling2D(2)(d2)  # 320x320
    if use_skip:
        u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    d1 = layers.Conv2D(64, 3, activation='relu', padding='same')(d1)

    # Output layer
    output = layers.Conv2D(out_ch, 3, padding='same')(d1)
    # If channels match
    if use_add:
        if input_shape[-1] == out_ch:
            output = layers.Add()([inputs, output])
        else:
            proj = layers.Conv2D(out_ch, 1, padding='same')(inputs)
            output = layers.Add()([proj, output])
        

    return tf.keras.Model(inputs, output)

#####################################################
def cg(H, b, x0=None, Nit=5, tol=0.0, eps=1e-15):
    """
    Solve H(x) = b with Conjugate Gradient.
    - H: callable, linear operator, e.g. lambda x: AHA_propeller(x, ...)
    - b: tf.Tensor (real or complex)
    - x0: initial guess (defaults to zeros_like(b))
    - Nit: max iterations
    - tol: relative residual tolerance (0.0 = fixed-iteration mode)
    """
    def c_inner(u, v):
        # complex-safe inner product -> real scalar
        return tf.math.real(tf.reduce_sum(tf.math.conj(u) * v))

    x = tf.zeros_like(b) if x0 is None else tf.identity(x0)
    r = b - H(x)
    p = tf.identity(r)
    rsold = c_inner(r, r)

    # initial residual norm (for relative stopping)
    r0 = tf.sqrt(tf.cast(rsold, tf.float32))

    for _ in range(Nit):
        Hp = H(p)
        denom = c_inner(p, Hp) + eps
        alpha = rsold / denom
        alpha_c = tf.cast(alpha, x.dtype)

        x = x + alpha_c * p
        r = r - alpha_c * Hp

        rsnew = c_inner(r, r)

        if tol > 0.0:
            rel = tf.sqrt(tf.cast(rsnew, tf.float32)) / (r0 + 1e-30)
            if float(rel) < tol:
                break

        beta = rsnew / (rsold + eps)
        beta_c = tf.cast(beta, x.dtype)

        p = r + beta_c * p
        rsold = rsnew

    return x
    
#####################################################################################

class ScalarParam(layers.Layer):
    """Trainable scalar with optional bounds; calling it scales its input."""
    def __init__(self, init=0.05, lb=None, ub=None, name="lam", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lb, self.ub = lb, ub
        self.init = float(init)

    def build(self, _):
        self.w = self.add_weight(
            name=f"{self.name}_w",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True, dtype=tf.float32
        )

    def value(self):
        v = self.w
        if self.lb is not None and self.ub is not None:
            v = self.lb + (self.ub - self.lb) * tf.sigmoid(v)  # keep in (lb,ub)
        return v

    def call(self, x):
        v = self.value()
        if x.dtype.is_complex:
            v = tf.complex(v, tf.zeros_like(v))
        return x * v
#####################################################################################

class UnrolledNet:
    def __init__(self, num_rows, num_cols, num_coils, num_blades, B1, B2):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_coils = num_coils
        self.num_blades = num_blades
        self.B1 = B1
        self.B2 = B2
        

        input_shapePC = (self.num_rows, self.num_cols, self.num_blades*2)
        input_shapeI = (self.num_rows, self.num_cols, 1)
        
        self.modelsPC = {}
        self.modelsI = {}

        for idx_iter in range(self.B1):
           self.modelsPC[str(idx_iter)]  = UNet(input_shape=input_shapePC, out_ch=num_blades*2, last_filt=1, use_skip=True, use_add=True)

        for idx_iter in range(self.B2):
          self.modelsI[str(idx_iter)] = UNet(input_shape=input_shapeI, out_ch=1, last_filt=1, use_skip=True, use_add=True)

    def build_model(self):
        # Inputs
        kdata_T = Input(shape=(self.num_coils,self.num_rows*self.num_cols, self.num_blades), dtype=tf.complex64)
        K_coord = Input(shape=(2,self.num_rows*self.num_cols, self.num_blades), dtype = tf.float32)
        csm = Input(shape=(self.num_coils, self.num_rows, self.num_cols), dtype=tf.complex64)
        kmask_T = Input(shape=(self.num_coils,self.num_rows*self.num_cols, self.num_blades), dtype=tf.complex64)

        GD_step_size1 = 0.4
        GD_step_size2 = 0.15

            
        # Compute right-hand side (A^H y)
        x = AH_propeller(kdata_T,K_coord,csm,kmask_T,self.num_blades)    
        
        xpc = x
        Ahb = x
        
        for idx_iter in range(self.B1):
            
          AhAx = AHA_propeller(x,K_coord,csm,kmask_T,self.num_blades)
          GD_grad = Subtract()([AhAx, Ahb])
          temp = Lambda(lambda y: y * GD_step_size1)(GD_grad)
          #temp = self.lam1(GD_grad) 

          x_GD = Subtract()([x, temp])

          x_real = complex_to_real(x_GD)
         # Run U-Nets
          img_PC = self.modelsPC[str(idx_iter)](x_real)
         
          x = real_to_complex(img_PC)
          
#################################################          
        output_LR = complex_to_real(x)
        epsilon = 1e-8
        phase_angle = tf.math.angle(x + tf.complex(epsilon, 0.0))
        PC = tf.exp(tf.complex(tf.zeros_like(x, dtype=tf.float32), phase_angle))
        
############### Phase constraint       
        x = tf.math.reduce_sum(tf.math.real(xpc*tf.math.conj(PC)),axis=-1,keepdims=True)
        Ahb = x
                 
        for idx_iter in range(self.B2):      
            
          xp = tf.tile(x, [1, 1, 1, self.num_blades])
          xp = tf.cast(xp, dtype=tf.complex64)
          xp = xp*PC#tf.where(is_finite_PC, xp * PC, xp)
          
          AhAx = AHA_propeller(xp,K_coord,csm,kmask_T,self.num_blades)
          
          AhAx = tf.math.reduce_sum(tf.math.real(AhAx*tf.math.conj(PC)),axis=-1,keepdims=True)
          GD_grad = Subtract()([AhAx, Ahb])
          temp = Lambda(lambda x: x * GD_step_size2)(GD_grad)
          #temp = self.lam2(GD_grad)
               
          
          x_GD = Subtract()([x, temp])

          # Run U-Nets
          x = self.modelsI[str(idx_iter)](x_GD)
 
## Consruct the complex-valued images        

        HR = tf.tile(x, [1, 1, 1, self.num_blades])
        HR = tf.cast(HR, dtype=tf.complex64)

        x = HR*PC#tf.where(is_finite_PC, HR * PC, HR)

        output_HR = complex_to_real(x)        
              

        # Final model
        model = Model(inputs=[kdata_T, K_coord ,csm, kmask_T], outputs=[output_HR, output_LR], name="UnrolledNet")
        return model  

###################################
num_rows = 320
num_cols = 320
num_coils = 8
num_blades = 20
num_batchs = 1
B1 = 1
B2 = 1

# ####################################################
# Angle = np.linspace(0, 180-1, 180)*np.pi/180
# Angle = Angle[0::int(180/num_blades)]

N=num_rows
im_size = [N,N]
grid_size = [2*N, 2*N]
nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho')
nufft_ob.numpoints = (5,5)
nufft_ob.numpoints_tensor = (5,5)

########################################################################################################################
input_paths = glob.glob(os.path.join(data_path, "Data_Train/*.mat"))
training_gen = DataGenerator(input_paths,num_rows, num_cols, num_coils, num_blades,num_batchs,shuffle=True)

####################################################################################
input_paths = glob.glob(os.path.join(data_path, "Data_Val/*.mat"))
val_gen = DataGenerator(input_paths,num_rows, num_cols, num_coils, num_blades,num_batchs,shuffle=True)

########################################################################################################################
tensorboard_filepath = os.path.join(data_path,'results')
model_checkpoint_filepath = os.path.join(data_path,'trained_weights', 'ZS_SS_MV_BLADE.h5')

# Define a learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4,  # Initial learning rate
    decay_steps=50,          # Number of steps before decay
    decay_rate=0.96,            # Factor by which the learning rate decays
    staircase=True)              # Apply discrete staircase decay

# Create the Adam optimizer with the learning rate schedule
adam_opt = Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6)

def my_objective_function(ref_kspace_tensor, nw_output_kspace):
    epsilon = tf.keras.backend.epsilon()
    scalar = tf.constant(0.5, dtype=tf.float32)

    l2_norm = tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor + epsilon)
    l1_norm = tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor + epsilon, ord=1)

    loss = scalar * l2_norm + scalar * l1_norm
    return loss

def custom_loss_fn_with_my_objective(num_blades):
    @tf.function
    def loss_fn(y_true, y_pred):
        kdata_L, K_coord ,csm, kmask_L = y_true
        x_hr, x_lr = y_pred

        x_hr = real_to_complex(x_hr)
        #x_lr = real_to_complex(x_lr)

        # Forward simulation
        output_kdata_hr = A_propeller(x_hr,K_coord,csm,kmask_L,num_blades)
        #output_kdata_lr = A_propeller(x_lr,K_coord,csm,kmask_L,num_blades)

        output_kdata_hr_real = tf.clip_by_value(complex_to_real(output_kdata_hr), -1.0, 1.0)
        #output_kdata_lr_real = tf.clip_by_value(complex_to_real(output_kdata_lr), -1.0, 1.0)


        return my_objective_function(kdata_L, output_kdata_hr_real)# + 0.5*my_objective_function(kdata_L, output_kdata_lr_real)

    return loss_fn

###############################################################################################################
loss_fn = custom_loss_fn_with_my_objective(num_blades)

unrolled_net = UnrolledNet(num_rows, num_cols, num_coils, num_blades, B1, B2)
base_model = unrolled_net.build_model()

class CustomUnrolledModel(tf.keras.Model):
    def __init__(self, base_model, loss_fn):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = loss_fn

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y = data
        kdata_T,K_coord,csm,kmask_T = x
        kdata_L,K_coord,csm,kmask_L = y

        with tf.GradientTape() as tape:
            y_pred = self([kdata_T,K_coord,csm,kmask_T], training=True)
            loss = self.loss_fn([kdata_L,K_coord,csm,kmask_L], y_pred)

        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        kdata_T,K_coord,csm,kmask_T = x
        kdata_L,K_coord,csm,kmask_L = y

        y_pred = self([kdata_T,K_coord,csm,kmask_T], training=False)
        loss = self.loss_fn([kdata_L,K_coord,csm,kmask_L], y_pred)

        return {"loss": loss}

model = CustomUnrolledModel(base_model, loss_fn)
model.compile(optimizer=adam_opt)
base_model.summary()

##########################################################################################################

# Build the full model by calling it once with dummy input
dummy_kdata = tf.zeros((1, num_coils, num_rows*num_cols, num_blades), dtype=tf.complex64)
dummy_K_coord = tf.zeros((1, 2, num_rows*num_cols, num_blades), dtype=tf.float32)
dummy_csm = tf.zeros((1, num_coils, num_rows, num_cols), dtype=tf.complex64)
dummy_mask = tf.zeros((1, num_coils, num_rows*num_cols, num_blades), dtype=tf.complex64)

_ = model([dummy_kdata, dummy_K_coord, dummy_csm, dummy_mask], training=False) 
##########################################################################################################
pre_train_path = None#os.path.join(data_path,'trained_weights', 'ZS_SS_MV_BLADE.h5')
if pre_train_path is not None:
  model.load_weights(pre_train_path)

# Callbacks

tbCallBack = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0, write_graph=False, write_images=False)
checkpointerCallBack = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')


hist = model.fit(x=training_gen,
                      epochs=180,
                      verbose=2,
                      callbacks=[tbCallBack,checkpointerCallBack],
                      validation_data=val_gen,
                      shuffle=True,
                      initial_epoch=0)


# ##
pre_train_path = os.path.join(data_path,'trained_weights', 'ZS_SS_MV_BLADE.h5')
model.load_weights(pre_train_path)

input_paths = glob.glob(os.path.join(data_path,'Data_Test/*.mat'))

t_gen = DataGenerator(input_paths,num_rows, num_cols, num_coils, num_blades,num_batchs,shuffle=True)

t=time.time()
recon_i = model.predict(t_gen)
elapsed = time.time()-t
print(elapsed)

im_cnn_hr= real_to_complex(np.array(recon_i[0]))
im_cnn_lr= real_to_complex(np.array(recon_i[1]))

# mI = np.mean(np.abs(im_cnn),axis=2)

mII = np.mean(np.abs(im_cnn_lr),axis=3)

plt.figure(figsize=(6, 6))
plt.imshow(np.fliplr(np.flipud(np.rot90(np.angle(im_cnn_lr[0,:, :,5]), k=-1))), cmap='gray')
plt.axis('off')  # optional: hide axis ticks
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(np.fliplr(np.flipud(np.rot90(np.abs(im_cnn_hr[0,:, :,0]), k=-1))), cmap='gray',vmin = 0.00, vmax = 0.55)
plt.axis('off')  # optional: hide axis ticks
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'],label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

Tloss = hist.history['loss']
Vloss = hist.history['val_loss']
    
    
sio.savemat(os.path.join(data_path, 'Recon_DL_V4/Final/ReconDL_Step015_AX_8Blades_10Splits_2B1_4B2_Slice_3.mat'),{'Im_HR':np.array(im_cnn_hr), 'Tloss':np.array(Tloss),'Vloss':np.array(Vloss)})

####################################################################


    
# ##################################################################  
# Nc = 8
# N = 320
# Np = 7

# input_paths = glob.glob(os.path.join(data_path,'Data_Test/*.mat'))

# K_coord = generate_blade_trajactory(N, 21)
# K_coord = K_coord[:, :, 0:21:3]
# K_coord = tf.convert_to_tensor(K_coord)[None,...]

# temp = sio.loadmat(input_paths[0])
# kdata_T = temp['kdata_T'].astype(np.csingle)
# kdata_T = kdata_T/np.max(np.abs(kdata_T))
# #kdata_T = tf.convert_to_tensor(kdata_T)[None,...]
        
# maskT = temp['mask_T'].astype(np.csingle)
# #maskT = tf.convert_to_tensor(maskT)[None,...]

# csm = temp['csm'].astype(np.csingle)
# csm = tf.convert_to_tensor(csm)[None,...]


# im_size = [N,N]
# grid_size = [2*N, 2*N]
# nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho')
# nufft_ob.numpoints = (5,5)
# nufft_ob.numpoints_tensor = (5,5)


# kdata_T = np.transpose(kdata_T,[2,0,1,3])
# kmask_T = np.transpose(maskT,[2,0,1,3])
# kdata_T = np.reshape(kdata_T,[Nc,N*N,Np])
# kdata_T = tf.convert_to_tensor(kdata_T)[None,...]
# kmask_T = np.reshape(kmask_T,[Nc,N*N,Np])
# kmask_T = tf.convert_to_tensor(kmask_T)[None,...]
        
# # --- setup (unchanged above) ---

# x0 = AH_propeller(kdata_T, K_coord, csm, kmask_T, Np)  # initial guess (can also start from zeros)
# Ahb = x0                                               # since AH_propeller(kdata_T, ...) already computed

# plt.figure(figsize=(10, 10))
# plt.subplot(121)
# plt.imshow(np.fliplr(np.flipud(np.rot90(np.abs(x[0,:, :,5]), k=-1))), cmap='gray')#,vmin = 0.01, vmax = 0.3)
# plt.axis('off')
# plt.show()


# # Define the linear operator H(x) = A^H A x
# def H(x):
#     return AHA_propeller(x, K_coord, csm, kmask_T, Np)

# # Complex-valued inner product <u,v> = sum(conj(u)*v) -> real scalar
# def c_inner(u, v):
#     return tf.math.real(tf.reduce_sum(tf.math.conj(u) * v))

# # --- Conjugate Gradient solver ---
# Nit = 10            # number of iterations (increase if needed)
# tol = 0.0           # fixed-iteration mode by default; set >0.0 for residual-based stopping

# x = tf.identity(x0)           # current estimate
# r = Ahb - H(x)                # initial residual
# p = tf.identity(r)            # initial search direction
# rsold = c_inner(r, r)         # ||r||^2 (real scalar)

# for k in range(Nit):
#     Hp = H(p)                                      # A^H A p
#     denom = c_inner(p, Hp) + 1e-15                 # avoid division by zero
#     alpha = rsold / denom                          # real scalar step
#     alpha_c = tf.cast(alpha, x.dtype)              # cast to complex for multiplication

#     x = x + alpha_c * p                            # update estimate
#     r = r - alpha_c * Hp                           # update residual

#     rsnew = c_inner(r, r)                          # new residual norm^2
#     if tol > 0.0 and float(rsnew) ** 0.5 < tol:    # optional early stop (eager mode)
#         break

#     beta = rsnew / (rsold + 1e-15)                 # CG coefficient (real scalar)
#     beta_c = tf.cast(beta, x.dtype)

#     p = r + beta_c * p                             # new search direction
#     rsold = rsnew

# # # --- visualize ---
# mII = np.mean(np.abs(x),axis=3)


# plt.figure(figsize=(10, 10))
# plt.subplot(121)
# plt.imshow(np.fliplr(np.flipud(np.rot90(np.abs(mII[0,:, :]), k=-1))), cmap='gray')#,vmin = 0.01, vmax = 0.3)
# plt.axis('off')
# plt.show()



# plt.figure(figsize=(6, 6))
# plt.imshow(np.fliplr(np.flipud(np.rot90(np.abs(x[0,:, :,7]), k=-1))), cmap='gray')#,vmin = 0.01, vmax = 0.3)
# plt.axis('off')  # optional: hide axis ticks
# plt.show()

# def generate_blade_trajactory(N, num_blades):

#     Angle = np.linspace(0, np.pi, num_blades,endpoint=False)
        
#     x = np.linspace(0, N-1, N)
#     y = np.linspace(0, N-1, N)
    
#     kx_base,ky_base = np.meshgrid(x,y)
    
#     kx_base=-(kx_base/N)+0.5;
#     ky_base=-(ky_base/N)+0.5;
    
#     k_traj = []
    
#     for it in range(num_blades):
        
#         angle = Angle[it]
    
#         xo=np.cos(angle)*kx_base+np.sin(angle)*ky_base;
#         yo=np.cos(angle)*ky_base-np.sin(angle)*kx_base;
    
#         kx = xo.flatten()
#         ky = yo.flatten()
#         ktraj = np.stack((-ky, -kx), axis=0)*2*np.pi
#     # convert k-space trajectory to a tensor and unsqueeze batch dimension
#         #ktraj = tf.convert_to_tensor(ktraj)[None, ...]
#     # create NUFFT objects, use 'ortho' for orthogonal FFTs
#         k_traj.append(ktraj)
    
#     k_traj = tf.stack(k_traj,axis=-1)
    
#     return k_traj



  