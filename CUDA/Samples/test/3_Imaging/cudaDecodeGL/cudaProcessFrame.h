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

#ifndef _CUDAPROCESSFRAME_H_
#define _CUDAPROCESSFRAME_H_

#include "dynlink_cuda.h" // <cuda.h>

typedef unsigned char   uint8;
typedef unsigned int    uint32;
typedef int             int32;

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff

typedef enum
{
    ITU601 = 1,
    ITU709 = 2
} eColorSpace;


// The NV12ToARGB helper functions and CUDA kernel launchers that get called
extern "C"
{
    CUresult InitCudaModule(char *filename_cubin, char *exec_path,   CUmodule   *cuModule);
    CUresult InitCudaFunction(char *func_name,      CUmodule *pModule, CUfunction *pCudaFunction);

    CUresult updateConstantMemory(float *hueCSC);
    CUresult updateConstantMemory_drvapi(CUmodule    cuModule, float *hueColorSpaceMat);
    void     setColorSpaceMatrix(eColorSpace      CSC, float *hueColorSpaceMat, float hue);

    CUresult cudaLaunchNV12toARGB(uint32 *d_srcNV12,        size_t nSourcePitch,
                                  uint32 *d_dstARGB,      size_t nDestPitch,
                                  uint32 width,           uint32 height);

    CUresult cudaLaunchNV12toARGB_4pix(uint32 *d_srcNV12,       size_t nSourcePitch,
                                       uint32 *d_dstARGB,      size_t nDestPitch,
                                       uint32 width,           uint32 height);

    CUresult cudaLaunchNV12toARGBDrv(CUdeviceptr d_srcNV12,   size_t nSourcePitch,
                                     CUdeviceptr d_dstARGB,  size_t nDestPitch,
                                     uint32 width,           uint32 height,
                                     CUfunction fpNV12toARGB, CUstream streamID);
};


#endif
