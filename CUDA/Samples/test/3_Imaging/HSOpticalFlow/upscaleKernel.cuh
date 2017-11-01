/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// scalar field to upscale
texture<float, 2, cudaReadModeElementType> texCoarse;

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = tex2D(texCoarse, x, y) * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static
void Upscale(const float *src, int width, int height, int stride,
             int newWidth, int newHeight, int newStride, float scale, float *out)
{
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

    // mirror if a coordinate value is out-of-range
    texCoarse.addressMode[0] = cudaAddressModeMirror;
    texCoarse.addressMode[1] = cudaAddressModeMirror;
    texCoarse.filterMode = cudaFilterModeLinear;
    texCoarse.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(0, texCoarse, src, width, height, stride * sizeof(float));

    UpscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, scale, out);
}
