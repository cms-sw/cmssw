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



#ifndef CONVOLUTIONFFT2D_COMMON_H
#define CONVOLUTIONFFT2D_COMMON_H



typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#endif


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

extern "C" void convolutionClampToBorderCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

extern "C" void padKernel(
    float *d_PaddedKernel,
    float *d_Kernel,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

extern "C" void padDataClampToBorder(
    float *d_PaddedData,
    float *d_Data,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

extern "C" void modulateAndNormalize(
    fComplex *d_Dst,
    fComplex *d_Src,
    int fftH,
    int fftW,
    int padding
);

extern "C" void spPostprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
);

extern "C" void spPreprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
);

extern "C" void spProcess2D(
    void *d_Data,
    void *d_Data0,
    void *d_Kernel0,
    uint DY,
    uint DX,
    int dir
);



#endif //CONVOLUTIONFFT2D_COMMON_H
