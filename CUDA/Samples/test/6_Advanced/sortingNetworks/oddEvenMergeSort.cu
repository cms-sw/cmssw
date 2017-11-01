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



//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm



#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"



////////////////////////////////////////////////////////////////////////////////
// Monolithic Bacther's sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for one or more small vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size <= arrayLength; size <<= 1)
    {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                Comparator(
                    s_key[pos - stride], s_val[pos - stride],
                    s_key[pos +      0], s_val[pos +      0],
                    dir
                );
        }
    }

    cg::sync(cta);
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Odd-even merge sort iteration kernel
// for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeGlobal(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;

    //Odd-even merge
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    if (stride < size / 2)
    {
        uint offset = global_comparatorI & ((size / 2) - 1);

        if (offset >= stride)
        {
            uint keyA = d_SrcKey[pos - stride];
            uint valA = d_SrcVal[pos - stride];
            uint keyB = d_SrcKey[pos +      0];
            uint valB = d_SrcVal[pos +      0];

            Comparator(
                keyA, valA,
                keyB, valB,
                dir
            );

            d_DstKey[pos - stride] = keyA;
            d_DstVal[pos - stride] = valA;
            d_DstKey[pos +      0] = keyB;
            d_DstVal[pos +      0] = valB;
        }
    }
    else
    {
        uint keyA = d_SrcKey[pos +      0];
        uint valA = d_SrcVal[pos +      0];
        uint keyB = d_SrcKey[pos + stride];
        uint valB = d_SrcVal[pos + stride];

        Comparator(
            keyA, valA,
            keyB, valB,
            dir
        );

        d_DstKey[pos +      0] = keyA;
        d_DstVal[pos +      0] = valA;
        d_DstKey[pos + stride] = keyB;
        d_DstVal[pos + stride] = valB;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function
extern "C" uint factorRadix2(uint *log2L, uint L);

extern "C" void oddEvenMergeSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint  blockCount = (batchSize * arrayLength) / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert(SHARED_SIZE_LIMIT % arrayLength == 0);
        oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
    {
        oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, SHARED_SIZE_LIMIT, dir);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                //Unlike with bitonic sort, combining bitonic merge steps with
                //stride = [SHARED_SIZE_LIMIT / 2 .. 1] seems to be impossible as there are
                //dependencies between data elements crossing the SHARED_SIZE_LIMIT borders
                oddEvenMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
            }
    }
}
