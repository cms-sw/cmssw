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

///////////////////////////////////////////////////////////////////////////////
#include <cufft.h>
#include <math_constants.h>

//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
    return (a + (b - 1)) / b;
}


// complex math functions
__device__
float2 conjugate(float2 arg)
{
    return make_float2(arg.x, -arg.y);
}

__device__
float2 complex_exp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

__device__
float2 complex_add(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__
float2 complex_mult(float2 ab, float2 cd)
{
    return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

// generate wave heightfield at time t based on initial heightfield and dispersion relationship
__global__ void generateSpectrumKernel(float2 *h0,
                                       float2 *ht,
                                       unsigned int in_width,
                                       unsigned int out_width,
                                       unsigned int out_height,
                                       float t,
                                       float patchSize)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int in_index = y*in_width+x;
    unsigned int in_mindex = (out_height - y)*in_width + (out_width - x); // mirrored
    unsigned int out_index = y*out_width+x;

    // calculate wave vector
    float2 k;
    k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
    k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

    // calculate dispersion w(k)
    float k_len = sqrtf(k.x*k.x + k.y*k.y);
    float w = sqrtf(9.81f * k_len);

    if ((x < out_width) && (y < out_height))
    {
        float2 h0_k = h0[in_index];
        float2 h0_mk = h0[in_mindex];

        // output frequency-space complex values
        ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
        //ht[out_index] = h0_k;
    }
}

// update height map values based on output of FFT
__global__ void updateHeightmapKernel(float  *heightMap,
                                      float2 *ht,
                                      unsigned int width)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = y*width+x;

    // cos(pi * (m1 + m2))
    float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

    heightMap[i] = ht[i].x * sign_correction;
}

// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = y*width+x;

    float2 slope = make_float2(0.0f, 0.0f);

    if ((x > 0) && (y > 0) && (x < width-1) && (y < height-1))
    {
        slope.x = h[i+1] - h[i-1];
        slope.y = h[i+width] - h[i-width];
    }

    slopeOut[i] = slope;
}

// wrapper functions
extern "C"
void cudaGenerateSpectrumKernel(float2 *d_h0,
                                float2 *d_ht,
                                unsigned int in_width,
                                unsigned int out_width,
                                unsigned int out_height,
                                float animTime,
                                float patchSize)
{
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y), 1);
    generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, in_width, out_width, out_height, animTime, patchSize);
}

extern "C"
void cudaUpdateHeightmapKernel(float  *d_heightMap,
                               float2 *d_ht,
                               unsigned int width,
                               unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_ht, width);
}

extern "C"
void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut,
                              unsigned int width, unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}
