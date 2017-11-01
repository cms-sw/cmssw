/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This example demonstrates how to use the Video Decode Library with CUDA
 * bindings to interop between CUDA and DX9 textures for the purpose of post
 * processing video.
 */

#include "cudaProcessFrame.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>

#include "dynlink_cuda.h" // <cuda.h>
#include "helper_cuda_drvapi.h"
#include "dynlink_builtin_types.h"

// These store the matrix for YUV2RGB transformation
__constant__ float  constHueColorSpaceMat[9];
__constant__ float  constAlpha;


extern "C"
CUresult  updateConstantMemory_drvapi(CUmodule module, float *hueCSC)
{
    CUdeviceptr  d_constHueCSC, d_constAlpha;
    size_t       d_cscBytes, d_alphaBytes;

    // First grab the global device pointers from the CUBIN
    cuModuleGetGlobal(&d_constHueCSC,  &d_cscBytes  , module, "constHueColorSpaceMat");
    cuModuleGetGlobal(&d_constAlpha ,  &d_alphaBytes, module, "constAlpha");

    CUresult error = CUDA_SUCCESS;

    // Copy the constants to video memory
    cuMemcpyHtoD(d_constHueCSC,
                 reinterpret_cast<const void *>(hueCSC),
                 d_cscBytes);
    getLastCudaDrvErrorMsg("cuMemcpyHtoD (d_constHueCSC) copy to Constant Memory failed");


    uint32 cudaAlpha      = ((uint32)0xff<< 24);

    cuMemcpyHtoD(d_constAlpha,
                 reinterpret_cast<const void *>(&cudaAlpha),
                 d_alphaBytes);
    getLastCudaDrvErrorMsg("cuMemcpyHtoD (constAlpha) copy to Constant Memory failed");

    return error;
}

extern "C"
void setColorSpaceMatrix(eColorSpace CSC, float *hueCSC, float hue)
{
    float hueSin = sin(hue);
    float hueCos = cos(hue);

    if (CSC == ITU601)
    {
        //CCIR 601
        hueCSC[0] = 1.1644f;
        hueCSC[1] = hueSin * 1.5960f;
        hueCSC[2] = hueCos * 1.5960f;
        hueCSC[3] = 1.1644f;
        hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
        hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
        hueCSC[6] = 1.1644f;
        hueCSC[7] = hueCos *  2.0172f;
        hueCSC[8] = hueSin * -2.0172f;
    }
    else if (CSC == ITU709)
    {
        //CCIR 709
        hueCSC[0] = 1.0f;
        hueCSC[1] = hueSin * 1.57480f;
        hueCSC[2] = hueCos * 1.57480f;
        hueCSC[3] = 1.0;
        hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
        hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
        hueCSC[6] = 1.0f;
        hueCSC[7] = hueCos *  1.85560f;
        hueCSC[8] = hueSin * -1.85560f;
    }
}

// We call this function to launch the CUDA kernel (NV12 to ARGB).
extern "C"
CUresult  cudaLaunchNV12toARGBDrv(CUdeviceptr d_srcNV12, size_t nSourcePitch,
                                  CUdeviceptr d_dstARGB, size_t nDestPitch,
                                  uint32 width,          uint32 height,
                                  CUfunction fpFunc, CUstream streamID)
{
    CUresult status;
    // Each thread will output 2 pixels at a time.  The grid size width is half
    // as large because of this
    dim3 block(32,16,1);
    dim3 grid((width+(2*block.x-1))/(2*block.x), (height+(block.y-1))/block.y, 1);

#if __CUDA_API_VERSION >= 4000
    // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
    void *args[] = { &d_srcNV12, &nSourcePitch,
                     &d_dstARGB, &nDestPitch,
                     &width, &height
                   };

    // new CUDA 4.0 Driver API Kernel launch call
    status = cuLaunchKernel(fpFunc, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, streamID,
                            args, NULL);
#else
    // This is the older Driver API launch method from CUDA (V1.0 to V3.2)
    checkCudaErrors(cuFuncSetBlockShape(fpFunc, block.x, block.y, 1));
    int offset = 0;

    // This method calls cuParamSetv() to pass device pointers also allows the ability to pass 64-bit device pointers

    // device pointer for Source Surface
    checkCudaErrors(cuParamSetv(fpFunc, offset, &d_srcNV12,    sizeof(d_srcNV12)));
    offset += sizeof(d_srcNV12);

    // set the Source pitch
    checkCudaErrors(cuParamSetv(fpFunc, offset, &nSourcePitch, sizeof(nSourcePitch)));
    offset += sizeof(nSourcePitch);

    // device pointer for Destination Surface
    checkCudaErrors(cuParamSetv(fpFunc, offset, &d_dstARGB,    sizeof(d_dstARGB)));
    offset += sizeof(d_dstARGB);

    //  set the Destination Pitch
    checkCudaErrors(cuParamSetv(fpFunc, offset, &nDestPitch,   sizeof(nDestPitch)));
    offset += sizeof(nDestPitch);

    // set the width of the image
    ALIGN_OFFSET(offset, __alignof(width));
    checkCudaErrors(cuParamSeti(fpFunc, offset, width));
    offset += sizeof(width);

    // set the height of the image
    ALIGN_OFFSET(offset, __alignof(height));
    checkCudaErrors(cuParamSeti(fpFunc, offset, height));
    offset += sizeof(height);

    checkCudaErrors(cuParamSetSize(fpFunc, offset));

    // Launching the kernel, we need to pass in the grid dimensions
    CUresult status = cuLaunchGridAsync(fpFunc, grid.x, grid.y, streamID);
#endif

    if (CUDA_SUCCESS != status)
    {
        fprintf(stderr, "cudaLaunchNV12toARGBDrv() failed to launch Kernel Function %p, retval = %d\n", fpFunc, status);
        return status;
    }

    return status;
}

