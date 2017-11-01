////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef _BOXFILTER_KERNEL_CH_
#define _BOXFILTER_KERNEL_CH_

#include <helper_math.h>
#include <helper_functions.h>

texture<float, 2> tex;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray *d_array, *d_tempArray;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
    Perform a fast box filter using the sliding window method.

    As the kernel moves from left to right, we add in the contribution of the new
    sample on the right, and subtract the value of the exiting sample on the left.
    This only requires 2 adds and a mul per output value, independent of the filter radius.
    The box filter is separable, so to perform a 2D box filter we perform the filter in
    the x direction, followed by the same filter in the y direction.
    Applying multiple iterations of the box filter converges towards a Gaussian blur.
    Using CUDA, rows or columns of the image are processed in parallel.
    This version duplicates edge pixels.

    Note that the x (row) pass suffers from uncoalesced global memory reads,
    since each thread is reading from a different row. For this reason it is
    better to use texture lookups for the x pass.
    The y (column) pass is perfectly coalesced.

    Parameters
    id - pointer to input data in global memory
    od - pointer to output data in global memory
    w  - image width
    h  - image height
    r  - filter radius

    e.g. for r = 2, w = 8:

    0 1 2 3 4 5 6 7
    x - -
    - x - -
    - - x - -
      - - x - -
        - - x - -
          - - x - -
            - - x -
              - - x
*/

// process row
__device__ void
d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void
d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void
d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void
d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

// texture version
// texture fetches automatically clamp to edge of image
__global__ void
d_boxfilter_x_tex(float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int x =- r; x <= r; x++)
    {
        t += tex2D(tex, x, y);
    }

    od[y * w] = t * scale;

    for (int x = 1; x < w; x++)
    {
        t += tex2D(tex, x + r, y);
        t -= tex2D(tex, x - r - 1, y);
        od[y * w + x] = t * scale;
    }
}

__global__ void
d_boxfilter_y_tex(float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int y = -r; y <= r; y++)
    {
        t += tex2D(tex, x, y);
    }

    od[x] = t * scale;

    for (int y = 1; y < h; y++)
    {
        t += tex2D(tex, x, y + r);
        t -= tex2D(tex, x, y - r - 1);
        od[y * w + x] = t * scale;
    }
}

// RGBA version
// reads from 32-bit unsigned int array holding 8-bit RGBA

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
           ((unsigned int)(rgba.z * 255.0f) << 16) |
           ((unsigned int)(rgba.y * 255.0f) <<  8) |
           ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

// row pass using texture lookups
__global__ void
d_boxfilter_rgba_x(unsigned int *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h)
    {
        float4 t = make_float4(0.0f);

        for (int x = -r; x <= r; x++)
        {
            t += tex2D(rgbaTex, x, y);
        }

        od[y * w] = rgbaFloatToInt(t * scale);

        for (int x = 1; x < w; x++)
        {
            t += tex2D(rgbaTex, x + r, y);
            t -= tex2D(rgbaTex, x - r - 1, y);
            od[y * w + x] = rgbaFloatToInt(t * scale);
        }
    }
}

// column pass using coalesced global memory reads
__global__ void
d_boxfilter_rgba_y(unsigned int *id, unsigned int *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    id = &id[x];
    od = &od[x];

    float scale = 1.0f / (float)((r << 1) + 1);

    float4 t;
    // do left edge
    t = rgbaIntToFloat(id[0]) * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += rgbaIntToFloat(id[y*w]);
    }

    od[0] = rgbaFloatToInt(t * scale);

    for (int y = 1; y < (r + 1); y++)
    {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[0]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += rgbaIntToFloat(id[(h - 1) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
}

extern "C"
void initTexture(int width, int height, void *pImage, bool useRGBA)
{
    int size = width * height * (useRGBA ? sizeof(uchar4) : sizeof(float));

    // copy image data to array
    cudaChannelFormatDesc channelDesc;
    if (useRGBA)
    {
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    }
    else
    {
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    }
    checkCudaErrors(cudaMallocArray(&d_array, &channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(d_array, 0, 0, pImage, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocArray(&d_tempArray,   &channelDesc, width, height));

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = true;

    // Bind the array to the texture
    if (useRGBA)
    {
        checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array, channelDesc));
    }
    else
    {
        checkCudaErrors(cudaBindTextureToArray(tex, d_array, channelDesc));
    }
}

extern "C"
void freeTextures()
{
    checkCudaErrors(cudaFreeArray(d_array));
    checkCudaErrors(cudaFreeArray(d_tempArray));
}


/*
    Perform 2D box filter on image using CUDA

    Parameters:
    d_src  - pointer to input image in device memory
    d_temp - pointer to temporary storage in device memory
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    radius - filter radius
    iterations - number of iterations

*/
extern "C"
double boxFilter(float *d_src, float *d_temp, float *d_dest, int width, int height,
                 int radius, int iterations, int nthreads, StopWatchInterface *timer)
{
    // var for kernel timing
    double dKernelTime = 0.0;

    // sync host and start computation timer_kernel
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTextureToArray(tex, d_array));

    for (int i=0; i<iterations; i++)
    {
        sdkResetTimer(&timer);
        // use texture for horizontal pass
        d_boxfilter_x_tex<<< height / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
        d_boxfilter_y_global<<< width / nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);

        // sync host and stop computation timer_kernel
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpyToArray(d_tempArray, 0, 0, d_dest, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTextureToArray(tex, d_tempArray));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

// RGBA version
extern "C"
double boxFilterRGBA(unsigned int *d_src, unsigned int *d_temp, unsigned int *d_dest, int width, int height,
                     int radius, int iterations, int nthreads, StopWatchInterface *timer)
{
    checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array));

    // var for kernel computation timing
    double dKernelTime;

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer_kernel
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        // use texture for horizontal pass
        d_boxfilter_rgba_x<<< height / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
        d_boxfilter_rgba_y<<< width / nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);

        // sync host and stop computation timer_kernel
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpyToArray(d_tempArray, 0, 0, d_dest, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_tempArray));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

#endif // #ifndef _BOXFILTER_KERNEL_H_
