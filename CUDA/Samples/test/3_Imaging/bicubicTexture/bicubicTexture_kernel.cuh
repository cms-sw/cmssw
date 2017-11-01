/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Bicubic filtering
    See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger
    http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html

    Reformulation thanks to Keenan Crane
*/

#ifndef _BICUBICTEXTURE_KERNEL_CUH_
#define _BICUBICTEXTURE_KERNEL_CUH_

enum Mode { MODE_NEAREST, MODE_BILINEAR, MODE_BICUBIC, MODE_FAST_BICUBIC, MODE_CATROM };

texture<uchar, 2, cudaReadModeNormalizedFloat> tex;
texture<uchar, 2, cudaReadModeElementType> tex2;    // need to use cudaReadModeElementType for tex2Dgather

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // texture data type, return type
__device__
R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                         );
}

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__
R tex2DFastBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    R r = g0(fy) * (g0x * tex2D(texref, px + h0x, py + h0y)   +
                    g1x * tex2D(texref, px + h1x, py + h0y)) +
          g1(fy) * (g0x * tex2D(texref, px + h0x, py + h1y)   +
                    g1x * tex2D(texref, px + h1x, py + h1y));
    return r;
}

// higher-precision 2D bilinear lookup
template<class T, class R>  // texture data type, return type
__device__
R tex2DBilinear(const texture<T, 2, cudaReadModeNormalizedFloat> tex, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;
    px += 0.5f;
    py += 0.5f;

    return lerp(lerp(tex2D(tex, px, py),        tex2D(tex, px + 1.0f, py), fx),
                lerp(tex2D(tex, px, py + 1.0f), tex2D(tex, px + 1.0f, py + 1.0f), fx), fy);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200

/*
    bilinear 2D texture lookup using tex2Dgather function
    - tex2Dgather() returns the four neighbouring samples in a single texture lookup
    - it is only supported on the Fermi architecture
    - you can select which component to fetch using the "comp" parameter
    - it can be used to efficiently implement custom texture filters

    The samples are returned in a 4-vector in the following order:
    x: (0, 1)
    y: (1, 1)
    z: (1, 0)
    w: (0, 0)
*/

template<class T, class R>  // texture data type, return type
__device__
float tex2DBilinearGather(const texture<T, 2, cudaReadModeElementType> texref, float x, float y, int comp=0)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;

    R samples = tex2Dgather(texref, px + 0.5f, py + 0.5f, comp);

    return lerp(lerp((float) samples.w, (float) samples.z, fx),
                lerp((float) samples.x, (float) samples.y, fx), fy);
}

#endif

// Catmull-Rom interpolation

__host__ __device__
float catrom_w0(float a)
{
    //return -0.5f*a + a*a - 0.5f*a*a*a;
    return a*(-0.5f + a*(1.0f - 0.5f*a));
}

__host__ __device__
float catrom_w1(float a)
{
    //return 1.0f - 2.5f*a*a + 1.5f*a*a*a;
    return 1.0f + a*a*(-2.5f + 1.5f*a);
}

__host__ __device__
float catrom_w2(float a)
{
    //return 0.5f*a + 2.0f*a*a - 1.5f*a*a*a;
    return a*(0.5f + a*(2.0f - 1.5f*a));
}

__host__ __device__
float catrom_w3(float a)
{
    //return -0.5f*a*a + 0.5f*a*a*a;
    return a*a*(-0.5f + 0.5f*a);
}

template<class T>
__device__
T catRomFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * catrom_w0(x);
    r += c1 * catrom_w1(x);
    r += c2 * catrom_w2(x);
    r += c3 * catrom_w3(x);
    return r;
}

// Note - can't use bilinear trick here because of negative lobes
template<class T, class R>  // texture data type, return type
__device__
R tex2DCatRom(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return catRomFilter<R>(fy,
                           catRomFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                          );
}


// test functions

// render image using normal bilinear texture lookup
__global__ void
d_render(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2D(tex, u, v);
        //float c = tex2DBilinear<uchar, float>(tex, u, v);
        //float c = tex2DBilinearGather<uchar, uchar4>(tex2, u, v, 0) / 255.0f;
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
    }
}

// render image using bicubic texture lookup
__global__ void
d_renderBicubic(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2DBicubic<uchar, float>(tex, u, v);
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
    }
}

// render image using fast bicubic texture lookup
__global__ void
d_renderFastBicubic(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2DFastBicubic<uchar, float>(tex, u, v);
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
    }
}

// render image using Catmull-Rom texture lookup
__global__ void
d_renderCatRom(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2DCatRom<uchar, float>(tex, u, v);
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
    }
}

#endif // _BICUBICTEXTURE_KERNEL_CUH_
