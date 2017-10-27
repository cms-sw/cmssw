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
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <device_functions.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile float *sdata, float mySum, const unsigned int tid, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid] = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    const int vid = tid & (VEC-1);

    float beta  = mySum;
    float temp;

    for (int i = VEC/2; i > 0; i>>=1)
    {
        if (vid < i)
        {
            temp      = sdata[tid+i];
            beta     += temp;
            sdata[tid]  = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) 
    {
        beta  = 0;
        for (int i = 0; i < blockDim.x; i += VEC) 
        {
            beta  += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void
reduceBlocks(const float *g_idata, float *g_odata, unsigned int n, cg::thread_block cta)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid, cta);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
}

// Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern float __shared__ smem[];

        // wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        cg::sync(cta);

        // The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            float mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid, cta);

            if (tid==0)
            {
                g_odata[0] = smem[0];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C"
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceMultiPass<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduceMultiPass<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduceMultiPass<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduceMultiPass< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduceMultiPass< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduceMultiPass< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduceMultiPass<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduceMultiPass<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduceMultiPass<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduceMultiPass<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceMultiPass<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduceMultiPass<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduceMultiPass<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduceMultiPass< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduceMultiPass< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduceMultiPass< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduceMultiPass<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduceMultiPass<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduceMultiPass<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduceMultiPass<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}

extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}

#endif // #ifndef _REDUCE_KERNEL_H_
