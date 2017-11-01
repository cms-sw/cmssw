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
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.
 */

// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <curand.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA Error handling

#include <cuda_runtime.h>
#include <curand.h>

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU);

const int    DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    printf("%s Starting...\n\n", argv[0]);

    // initialize the GPU, either identified by --device
    // or by picking the device with highest flop rate.
    int devID = findCudaDevice(argc, (const char **)argv);

    // parsing the number of random numbers to generate
    int rand_n = DEFAULT_RAND_N;

    if (checkCmdLineFlag(argc, (const char **) argv, "count"))
    {
        rand_n = getCmdLineArgumentInt(argc, (const char **) argv, "count");
    }

    printf("Allocating data for %i samples...\n", rand_n);

    // parsing the seed
    int seed = DEFAULT_SEED;

    if (checkCmdLineFlag(argc, (const char **) argv, "seed"))
    {
        seed = getCmdLineArgumentInt(argc, (const char **) argv, "seed");
    }

    printf("Seeding with %i ...\n", seed);


    float *d_Rand;
    checkCudaErrors(cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)));

    curandGenerator_t prngGPU;
    checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));

    curandGenerator_t prngCPU;
    checkCudaErrors(curandCreateGeneratorHost(&prngCPU, CURAND_RNG_PSEUDO_MTGP32));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngCPU, seed));

    //
    // Example 1: Compare random numbers generated on GPU and CPU
    float *h_RandGPU  = (float *)malloc(rand_n * sizeof(float));

    printf("Generating random numbers on GPU...\n\n");
    checkCudaErrors(curandGenerateUniform(prngGPU, (float *) d_Rand, rand_n));

    printf("\nReading back the results...\n");
    checkCudaErrors(cudaMemcpy(h_RandGPU, d_Rand, rand_n * sizeof(float), cudaMemcpyDeviceToHost));


    float *h_RandCPU  = (float *)malloc(rand_n * sizeof(float));

    printf("Generating random numbers on CPU...\n\n");
    checkCudaErrors(curandGenerateUniform(prngCPU, (float *) h_RandCPU, rand_n));

    printf("Comparing CPU/GPU random numbers...\n\n");
    float L1norm = compareResults(rand_n, h_RandGPU, h_RandCPU);

    //
    // Example 2: Timing of random number generation on GPU
    const int numIterations = 10;
    int i;
    StopWatchInterface *hTimer;

    checkCudaErrors(cudaDeviceSynchronize());
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < numIterations; i++)
    {
        checkCudaErrors(curandGenerateUniform(prngGPU, (float *) d_Rand, rand_n));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

    double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer)/(double)numIterations;

    printf("MersenneTwisterGP11213, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers\n",
           1.0e-9 * rand_n / gpuTime, gpuTime, rand_n);

    printf("Shutting down...\n");

    checkCudaErrors(curandDestroyGenerator(prngGPU));
    checkCudaErrors(curandDestroyGenerator(prngCPU));
    checkCudaErrors(cudaFree(d_Rand));
    sdkDeleteTimer(&hTimer);
    free(h_RandGPU);
    free(h_RandCPU);

    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}


float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU)
{
    int i;
    float rCPU, rGPU, delta;
    float max_delta = 0.;
    float sum_delta = 0.;
    float sum_ref   = 0.;

    for (i = 0; i < rand_n; i++)
    {
        rCPU = h_RandCPU[i];
        rGPU = h_RandGPU[i];
        delta = fabs(rCPU - rGPU);
        sum_delta += delta;
        sum_ref   += fabs(rCPU);

        if (delta >= max_delta)
        {
            max_delta = delta;
        }
    }

    float L1norm = (float)(sum_delta / sum_ref);
    printf("Max absolute error: %E\n", max_delta);
    printf("L1 norm: %E\n\n", L1norm);

    return L1norm;
}
