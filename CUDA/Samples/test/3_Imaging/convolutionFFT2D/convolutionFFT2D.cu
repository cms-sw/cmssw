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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolutionFFT2D_common.h"
#include "convolutionFFT2D.cuh"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

    SET_FLOAT_BASE;
    padKernel_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

    SET_FLOAT_BASE;
    padDataClampToBorder_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(
    fComplex *d_Dst,
    fComplex *d_Src,
    int fftH,
    int fftW,
    int padding
)
{
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);

    modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
        d_Dst,
        d_Src,
        dataSize,
        1.0f / (float)(fftW *fftH)
    );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
static const double PI = 3.1415926535897932384626433832795;
static const uint BLOCKDIM = 256;

extern "C" void spPostprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = dir * PI / (double)DX;

    SET_FCOMPLEX_BASE;
    spPostprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
    );
    getLastCudaError("spPostprocess2D_kernel<<<>>> execution failed\n");
}

extern "C" void spPreprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = -dir * PI / (double)DX;

    SET_FCOMPLEX_BASE;
    spPreprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
    );
    getLastCudaError("spPreprocess2D_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
extern "C" void spProcess2D(
    void *d_Dst,
    void *d_SrcA,
    void *d_SrcB,
    uint DY,
    uint DX,
    int dir
)
{
    assert(DY % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = (DY / 2) * DX;
    const double phaseBase = dir * PI / (double)DX;

    SET_FCOMPLEX_BASE_A;
    SET_FCOMPLEX_BASE_B;
    spProcess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_SrcA,
        (fComplex *)d_SrcB,
        DY, DX, threadCount,
        (float)phaseBase,
        0.5f / (float)(DY *DX)
    );
    getLastCudaError("spProcess2D_kernel<<<>>> execution failed\n");
}
