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

#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void
reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduce1(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version uses the warp shuffle operation if available to reduce 
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    See http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    for additional information about using shuffle to perform a reduction
    within a warp.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce4(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += active.shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    cg::sync(cta);
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version is completely unrolled, unless warp shuffle is available, then
    shuffle is used within a loop.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time. When shuffle is available, it is used to reduce warp synchronization.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce5(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += active.shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    cg::sync(cta);
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

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

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
             mySum += active.shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    cg::sync(cta);
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}


extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 0:
            reduce0<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 1:
            reduce1<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 2:
            reduce2<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 3:
            reduce3<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 4:
            switch (threads)
            {
                case 512:
                    reduce4<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 256:
                    reduce4<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 128:
                    reduce4<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 64:
                    reduce4<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 32:
                    reduce4<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 16:
                    reduce4<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  8:
                    reduce4<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  4:
                    reduce4<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  2:
                    reduce4<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  1:
                    reduce4<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;
            }

            break;

        case 5:
            switch (threads)
            {
                case 512:
                    reduce5<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 256:
                    reduce5<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 128:
                    reduce5<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 64:
                    reduce5<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 32:
                    reduce5<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 16:
                    reduce5<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  8:
                    reduce5<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  4:
                    reduce5<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  2:
                    reduce5<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case  1:
                    reduce5<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;
            }

            break;

        case 6:
        default:
            if (isPow2(size))
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }
            else
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }

            break;
    }
}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);

template void
reduce<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
reduce<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);


#endif // #ifndef _REDUCE_KERNEL_H_
