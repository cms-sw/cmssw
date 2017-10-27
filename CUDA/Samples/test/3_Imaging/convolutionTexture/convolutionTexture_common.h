/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CONVOLUTIONTEXTURE_COMMON_H
#define CONVOLUTIONTEXTURE_COMMON_H



#include <cuda_runtime.h>



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel size (the only parameter inlined in the code)
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)



////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);



////////////////////////////////////////////////////////////////////////////////
// GPU texture-based convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel(float *h_Kernel);

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);



#endif
