/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A program demonstrating trivial use of system-wide atomics on migratable memory.
 */
 
#include <math.h>
#include <cstdio>
#include <ctime>
#include <stdint.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)

#define LOOP_NUM 50

 __global__ void atomicKernel(int *atom_arr)
 {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i=0; i < LOOP_NUM; i++)
    {
        // Atomic addition
        atomicAdd_system(&atom_arr[0], 10);

        // Atomic exchange
        atomicExch_system(&atom_arr[1], tid);

        // Atomic maximum
        atomicMax_system(&atom_arr[2], tid);

        // Atomic minimum
        atomicMin_system(&atom_arr[3], tid);

        // Atomic increment (modulo 17+1)
        atomicInc_system((unsigned int *)&atom_arr[4], 17);

        // Atomic decrement
        atomicDec_system((unsigned int *)&atom_arr[5], 137);

        // Atomic compare-and-swap
        atomicCAS_system(&atom_arr[6], tid-1, tid);

        // Bitwise atomic instructions

        // Atomic AND
        atomicAnd_system(&atom_arr[7], 2*tid+7);

        // Atomic OR
        atomicOr_system(&atom_arr[8], 1 << tid);

        // Atomic XOR
        atomicXor_system(&atom_arr[9], tid);
    }
 }

void atomicKernel_CPU(int *atom_arr, int no_of_threads)
{

    for (int i=no_of_threads; i<2*no_of_threads; i++)
    {

        for (int j=0; j < LOOP_NUM; j++)
        {
            // Atomic addition
            __sync_fetch_and_add(&atom_arr[0],10);

            // Atomic exchange
             __sync_lock_test_and_set(&atom_arr[1], i);

            // Atomic maximum
            int old, expected;
            do {
                expected = atom_arr[2];
                old = __sync_val_compare_and_swap(&atom_arr[2], expected, max(expected, i));
            } while (old != expected);

            // Atomic minimum
            do {
                expected = atom_arr[3];
                old = __sync_val_compare_and_swap(&atom_arr[3], expected, min(expected, i));
            } while (old != expected);

            // Atomic increment (modulo 17+1)
            int limit = 17;
            do {
                expected = atom_arr[4];
                old = __sync_val_compare_and_swap(&atom_arr[4], expected, (expected >= limit) ? 0 : expected+1);
            } while (old != expected);

            // Atomic decrement
            limit = 137;
            do {
                expected = atom_arr[5];
                old = __sync_val_compare_and_swap(&atom_arr[5], expected, ((expected == 0) || (expected > limit)) ? limit : expected-1);
            } while (old != expected);

            // Atomic compare-and-swap
           __sync_val_compare_and_swap(&atom_arr[6], i-1, i);


            // Bitwise atomic instructions

            // Atomic AND
             __sync_fetch_and_and(&atom_arr[7] , 2 * i + 7);

            // Atomic OR
             __sync_fetch_and_or(&atom_arr[8], 1 << i);

            // Atomic XOR
             // 11th element should be 0xff
            __sync_fetch_and_xor (&atom_arr[9] ,i);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int verify(int *testData, const int len)
{
    int val = 0;

    for (int i = 0; i < len*LOOP_NUM; ++i)
    {
        val += 10;
    }

    if (val != testData[0])
    {
        printf("atomicAdd failed val = %d testData = %d\n", val, testData[0]);
        return false;
    }

    val = 0;

    bool found = false;

    for (int i = 0; i < len; ++i)
    {
        // second element should be a member of [0, len)
        if (i == testData[1])
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        printf("atomicExch failed\n");
        return false;
    }

    val = -(1 << 8);

    for (int i = 0; i < len; ++i)
    {
        // third element should be len-1
        val = max(val, i);
    }

    if (val != testData[2])
    {
        printf("atomicMax failed\n");
        return false;
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, i);
    }

    if (val != testData[3])
    {
        printf("atomicMin failed\n");
        return false;
    }

    int limit = 17;
    val = 0;

    for (int i = 0; i < len*LOOP_NUM; ++i)
    {
        val = (val >= limit) ? 0 : val+1;
    }

    if (val != testData[4])
    {
        printf("atomicInc failed\n");
        return false;
    }

    limit = 137;
    val = 0;

    for (int i = 0; i < len*LOOP_NUM; ++i)
    {
        val = ((val == 0) || (val > limit)) ? limit : val-1;
    }

    if (val != testData[5])
    {
        printf("atomicDec failed\n");
        return false;
    }

    found = false;

    for (int i = 0; i < len; ++i)
    {
        // seventh element should be a member of [0, len)
        if (i == testData[6])
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        printf("atomicCAS failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 8th element should be 1
        val &= (2 * i + 7);
    }

    if (val != testData[7])
    {
        printf("atomicAnd failed\n");
        return false;
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        // 9th element should be 0xff
        val |= (1 << i);
    }

    if (val != testData[8])
    {
        printf("atomicOr failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 11th element should be 0xff
        val ^= i;
    }

    if (val != testData[9])
    {
        printf("atomicXor failed\n");
        return false;
    }

    return true;
}


int main(int argc, char **argv)
{
    // set device
    cudaDeviceProp device_prop;
    int dev_id = findCudaDevice(argc, (const char **) argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

    if (!device_prop.managedMemory) {
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    if (device_prop.computeMode == cudaComputeModeProhibited)
    {
        // This sample requires being run with a default or process exclusive mode
        fprintf(stderr, "This sample requires a device in either default or process exclusive mode\n");
        exit(EXIT_WAIVED);
    }

    if (device_prop.major < 6)
    {
        printf("%s: requires a minimum CUDA compute 6.0 capability, waiving testing.\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 10;

    int *atom_arr;

    if (device_prop.pageableMemoryAccess)
    {
        printf("CAN access pageable memory\n");
        atom_arr = (int *) malloc(sizeof(int)*numData);
    }
    else
    {
        printf("CANNOT access pageable memory\n");
        checkCudaErrors(cudaMallocManaged(&atom_arr, sizeof(int)*numData));
    }

    for (unsigned int i = 0; i < numData; i++)
        atom_arr[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    atom_arr[7] = atom_arr[9] = 0xff;

    atomicKernel<<<numBlocks, numThreads>>>(atom_arr);
    atomicKernel_CPU(atom_arr, numBlocks*numThreads);

    checkCudaErrors(cudaDeviceSynchronize());

    // Compute & verify reference solution
    int testResult = verify(atom_arr, 2*numThreads*numBlocks);

    if (device_prop.pageableMemoryAccess)
    {
        free(atom_arr);
    }
    else
    {
        cudaFree(atom_arr);
    }

    printf("systemWideAtomics completed, returned %s \n", testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
