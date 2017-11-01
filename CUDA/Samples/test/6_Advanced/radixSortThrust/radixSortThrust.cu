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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

template <typename T, bool floatKeys>
bool testSort(int argc, char **argv)
{
    int cmdVal;
    int keybits = 32;

    unsigned int numElements = 1048576;
    bool keysOnly = checkCmdLineFlag(argc, (const char **)argv, "keysonly");
    bool quiet    = checkCmdLineFlag(argc, (const char **)argv, "quiet");

    if (checkCmdLineFlag(argc, (const char **)argv, "n"))
    {
        cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        numElements = cmdVal;

        if (cmdVal < 0)
        {
            printf("Error: elements must be > 0, elements=%d is invalid\n", cmdVal);
            exit(EXIT_SUCCESS);
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "keybits"))
    {
        cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "keybits");
        keybits = cmdVal;

        if (keybits <= 0)
        {
            printf("Error: keybits must be > 0, keybits=%d is invalid\n", keybits);
            exit(EXIT_SUCCESS);
        }
    }

    unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))
    {
        cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
        numIterations = cmdVal;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Command line:\nradixSortThrust [-option]\n");
        printf("Valid options:\n");
        printf("-n=<N>        : number of elements to sort\n");
        printf("-keybits=bits : keybits must be > 0\n");
        printf("-keysonly     : only sort an array of keys (default sorts key-value pairs)\n");
        printf("-float        : use 32-bit float keys (default is 32-bit unsigned int)\n");
        printf("-quiet        : Output only the number of elements and the time to sort\n");
        printf("-help         : Output a help message\n");
        exit(EXIT_SUCCESS);
    }

    if (!quiet)
        printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits, floatKeys ? "float" : "unsigned int", keysOnly ? "(only)" : "and values");

    int deviceID = -1;

    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = (keysOnly ? 2 : 4) * numElements * sizeof(T);

        if (devprop.totalGlobalMem < totalMem)
        {
            printf("Error: insufficient amount of memory to sort %d elements.\n", numElements);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(EXIT_SUCCESS);
        }
    }

    thrust::host_vector<T> h_keys(numElements);
    thrust::host_vector<T> h_keysSorted(numElements);
    thrust::host_vector<unsigned int> h_values;

    if (!keysOnly)
        h_values = thrust::host_vector<unsigned int>(numElements);

    // Fill up with some random data
    thrust::default_random_engine rng(clock());

    if (floatKeys)
    {
        thrust::uniform_real_distribution<float> u01(0, 1);

        for (int i = 0; i < (int)numElements; i++)
            h_keys[i] = u01(rng);
    }
    else
    {
        thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

        for (int i = 0; i < (int)numElements; i++)
            h_keys[i] = u(rng);
    }

    if (!keysOnly)
        thrust::sequence(h_values.begin(), h_values.end());

    // Copy data onto the GPU
    thrust::device_vector<T> d_keys;
    thrust::device_vector<unsigned int> d_values;

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    float totalTime = 0;

    for (unsigned int i = 0; i < numIterations; i++)
    {
        // reset data before sort
        d_keys= h_keys;

        if (!keysOnly)
            d_values = h_values;

        checkCudaErrors(cudaEventRecord(start_event, 0));

        if (keysOnly)
            thrust::sort(d_keys.begin(), d_keys.end());
        else
            thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

        float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    totalTime /= (1.0e3f * numIterations);
    printf("radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-6f * numElements / totalTime, totalTime, numElements);

    getLastCudaError("after radixsort");

    // Get results back to host for correctness checking
    thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

    if (!keysOnly)
        thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

    getLastCudaError("copying results to host memory");

    // Check results
    bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    if (!bTestResult  && !quiet)
    {
        return false;
    }

    return bTestResult;
}

int main(int argc, char **argv)
{
    // Start logs
    printf("%s Starting...\n\n", argv[0]);

    findCudaDevice(argc, (const char **)argv);

    bool bTestResult = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "float"))
        bTestResult = testSort<float, true>(argc, argv);
    else
        bTestResult = testSort<unsigned int, false>(argc, argv);

    printf(bTestResult ? "Test passed\n" : "Test failed!\n");
}

