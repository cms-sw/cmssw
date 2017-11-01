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

/// image to downscale
texture<float, 2, cudaReadModeElementType> texFine;

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void DownscaleKernel(int width, int height, int stride, float *out)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
    {
        return;
    }

    float dx = 1.0f/(float)width;
    float dy = 1.0f/(float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    out[ix + iy * stride] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
                                     tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static
void Downscale(const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out)
{
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

    // mirror if a coordinate value is out-of-range
    texFine.addressMode[0] = cudaAddressModeMirror;
    texFine.addressMode[1] = cudaAddressModeMirror;
    texFine.filterMode = cudaFilterModeLinear;
    texFine.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

    DownscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, out);
}
