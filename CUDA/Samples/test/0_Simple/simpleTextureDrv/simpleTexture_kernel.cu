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

#ifndef _SIMPLETEXTURE_KERNEL_H_
#define _SIMPLETEXTURE_KERNEL_H_

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
transformKernel(float *g_odata, int width, int height, float theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = (float)x - (float)width/2; 
    float v = (float)y - (float)height/2; 
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 

    tu /= (float)width; 
    tv /= (float)height; 

    // read from texture and write to global memory
    g_odata[y*width + x] = tex2D(tex, tu+0.5f, tv+0.5f);
}

#endif // #ifndef _SIMPLETEXTURE_KERNEL_H_
