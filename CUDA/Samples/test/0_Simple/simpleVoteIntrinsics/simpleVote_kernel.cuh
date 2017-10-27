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

#ifndef SIMPLEVOTE_KERNEL_CU
#define SIMPLEVOTE_KERNEL_CU

////////////////////////////////////////////////////////////////////////////////
// Vote Any/All intrinsic kernel function tests are supported only by CUDA
// capable devices that are CUDA hardware that has SM1.2 or later
// Vote Functions (refer to section 4.4.5 in the CUDA Programming Guide)
////////////////////////////////////////////////////////////////////////////////

// Kernel #1 tests the across-the-warp vote(any) intrinsic.
// If ANY one of the threads (within the warp) of the predicated condition returns
// a non-zero value, then all threads within this warp will return a non-zero value
__global__ void VoteAnyKernel1(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask = 0xffffffff;
    result[tx] = __any_sync(mask, input[tx]);
}

// Kernel #2 tests the across-the-warp vote(all) intrinsic.
// If ALL of the threads (within the warp) of the predicated condition returns
// a non-zero value, then all threads within this warp will return a non-zero value
__global__ void VoteAllKernel2(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask = 0xffffffff;
    result[tx] = __all_sync(mask, input[tx]);
}

// Kernel #3 is a directed test for the across-the-warp vote(all) intrinsic.
// This kernel will test for conditions across warps, and within half warps
__global__ void VoteAnyKernel3(bool *info, int warp_size)
{
    int tx = threadIdx.x;
    unsigned int mask = 0xffffffff;
    bool *offs = info + (tx * 3);

    // The following should hold true for the second and third warp
    *offs = __any_sync(mask, (tx >= (warp_size * 3) / 2));
    // The following should hold true for the "upper half" of the second warp,
    // and all of the third warp
    *(offs + 1) = (tx >= (warp_size * 3) / 2? true: false);

    // The following should hold true for the third warp only
    if (all((tx >= (warp_size * 3) / 2)))
    {
        *(offs + 2) = true;
    }
}

#endif
