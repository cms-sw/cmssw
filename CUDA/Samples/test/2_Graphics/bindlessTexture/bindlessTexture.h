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

#ifndef _BINDLESSTEXTURE_CU_
#define _BINDLESSTEXTURE_CU_

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <vector_types.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

#pragma pack(push,4)
struct Image
{
    void                   *h_data;
    cudaExtent              size;
    cudaResourceType        type;
    cudaArray_t             dataArray;
    cudaMipmappedArray_t    mipmapArray;
    cudaTextureObject_t     textureObject;

    Image()
    {
        memset(this,0,sizeof(Image));
    }
};
#pragma pack(pop)

inline void _checkHost(bool test, const char *condition, const char *file, int line, const char *func)
{
    if (!test)
    {
        fprintf(stderr, "HOST error at %s:%d (%s) \"%s\" \n",
                file, line, condition, func);
        exit(EXIT_FAILURE);
    }
}

#define checkHost(condition)   _checkHost(condition, #condition,__FILE__,__LINE__,__FUNCTION__)

#endif

