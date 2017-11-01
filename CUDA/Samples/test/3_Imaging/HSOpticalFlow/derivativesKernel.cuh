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

/// source image
texture<float, 2, cudaReadModeElementType> texSource;
/// tracked image
texture<float, 2, cudaReadModeElementType> texTarget;

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height) return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    float t0, t1;
    // x derivative
    t0  = tex2D(texSource, x - 2.0f * dx, y);
    t0 -= tex2D(texSource, x - 1.0f * dx, y) * 8.0f;
    t0 += tex2D(texSource, x + 1.0f * dx, y) * 8.0f;
    t0 -= tex2D(texSource, x + 2.0f * dx, y);
    t0 /= 12.0f;

    t1  = tex2D(texTarget, x - 2.0f * dx, y);
    t1 -= tex2D(texTarget, x - 1.0f * dx, y) * 8.0f;
    t1 += tex2D(texTarget, x + 1.0f * dx, y) * 8.0f;
    t1 -= tex2D(texTarget, x + 2.0f * dx, y);
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // t derivative
    Iz[pos] = tex2D(texTarget, x, y) - tex2D(texSource, x, y);

    // y derivative
    t0  = tex2D(texSource, x, y - 2.0f * dy);
    t0 -= tex2D(texSource, x, y - 1.0f * dy) * 8.0f;
    t0 += tex2D(texSource, x, y + 1.0f * dy) * 8.0f;
    t0 -= tex2D(texSource, x, y + 2.0f * dy);
    t0 /= 12.0f;

    t1  = tex2D(texTarget, x, y - 2.0f * dy);
    t1 -= tex2D(texTarget, x, y - 1.0f * dy) * 8.0f;
    t1 += tex2D(texTarget, x, y + 1.0f * dy) * 8.0f;
    t1 -= tex2D(texTarget, x, y + 2.0f * dy);
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static
void ComputeDerivatives(const float *I0, const float *I1,
                        int w, int h, int s,
                        float *Ix, float *Iy, float *Iz)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    // mirror if a coordinate value is out-of-range
    texSource.addressMode[0] = cudaAddressModeMirror;
    texSource.addressMode[1] = cudaAddressModeMirror;
    texSource.filterMode = cudaFilterModeLinear;
    texSource.normalized = true;

    texTarget.addressMode[0] = cudaAddressModeMirror;
    texTarget.addressMode[1] = cudaAddressModeMirror;
    texTarget.filterMode = cudaFilterModeLinear;
    texTarget.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(0, texSource, I0, w, h, s * sizeof(float));
    cudaBindTexture2D(0, texTarget, I1, w, h, s * sizeof(float));

    ComputeDerivativesKernel<<<blocks, threads>>>(w, h, s, Ix, Iy, Iz);
}
