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

/*
    Recursive Gaussian filter
    sgreen 8/1/08

    This code sample implements a Gaussian blur using Deriche's recursive method:
    http://citeseer.ist.psu.edu/deriche93recursively.html

    This is similar to the box filter sample in the SDK, but it uses the previous
    outputs of the filter as well as the previous inputs. This is also known as an
    IIR (infinite impulse response) filter, since its response to an input impulse
    can last forever.

    The main advantage of this method is that the execution time is independent of
    the filter width.

    The GPU processes columns of the image in parallel. To avoid uncoalesced reads
    for the row pass we transpose the image and then transpose it back again
    afterwards.

    The implementation is based on code from the CImg library:
    http://cimg.sourceforge.net/
    Thanks to David Tschumperl� and all the CImg contributors!
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "recursiveGaussian_kernel.cuh"

#define USE_SIMPLE_FILTER 0

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*
    Transpose a 2D array (see SDK transpose example)
*/
extern "C"
void transpose(uint *d_src, uint *d_dest, uint width, int height)
{
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose<<< grid, threads >>>(d_dest, d_src, width, height);
    getLastCudaError("Kernel execution failed");
}

/*
    Perform Gaussian filter on a 2D image using CUDA

    Parameters:
    d_src  - pointer to input image in device memory
    d_dest - pointer to destination image in device memory
    d_temp - pointer to temporary storage in device memory
    width  - image width
    height - image height
    sigma  - sigma of Gaussian
    order  - filter order (0, 1 or 2)
*/

// 8-bit RGBA version
extern "C"
void gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads)
{
    // compute filter coefficients
    const float
    nsigma = sigma < 0.1f ? 0.1f : sigma,
    alpha = 1.695f / nsigma,
    ema = (float)std::exp(-alpha),
    ema2 = (float)std::exp(-2*alpha),
    b1 = -2*ema,
    b2 = ema2;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;

    switch (order)
    {
        case 0:
            {
                const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
                a0 = k;
                a1 = k*(alpha-1)*ema;
                a2 = k*(alpha+1)*ema;
                a3 = -k*ema2;
            }
            break;

        case 1:
            {
                const float k = (1-ema)*(1-ema)/ema;
                a0 = k*ema;
                a1 = a3 = 0;
                a2 = -a0;
            }
            break;

        case 2:
            {
                const float
                ea = (float)std::exp(-alpha),
                k = -(ema2-1)/(2*alpha*ema),
                kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
                a0 = kn;
                a1 = -kn*(1+k*alpha)*ema;
                a2 = kn*(1-k*alpha)*ema;
                a3 = -kn*ema2;
            }
            break;

        default:
            fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
            return;
    }

    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);

    // process columns
#if USE_SIMPLE_FILTER
    d_simpleRecursive_rgba<<< iDivUp(width, nthreads), nthreads >>>(d_src, d_temp, width, height, ema);
#else
    d_recursiveGaussian_rgba<<< iDivUp(width, nthreads), nthreads >>>(d_src, d_temp, width, height, a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
    getLastCudaError("Kernel execution failed");

    transpose(d_temp, d_dest, width, height);
    getLastCudaError("transpose: Kernel execution failed");

    // process rows
#if USE_SIMPLE_FILTER
    d_simpleRecursive_rgba<<< iDivUp(height, nthreads), nthreads >>>(d_dest, d_temp, height, width, ema);
#else
    d_recursiveGaussian_rgba<<< iDivUp(height, nthreads), nthreads >>>(d_dest, d_temp, height, width, a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
    getLastCudaError("Kernel execution failed");

    transpose(d_temp, d_dest, height, width);
}
