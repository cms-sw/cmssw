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
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "binomialOptions_common.h"
#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsCPU(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
real randData(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);

    int devID = findCudaDevice(argc, (const char **)argv);

    const int OPT_N = MAX_OPTIONS;

    TOptionData optionData[MAX_OPTIONS];
    real
    callValueBS[MAX_OPTIONS],
                callValueGPU[MAX_OPTIONS],
                callValueCPU[MAX_OPTIONS];

    real
    sumDelta, sumRef, gpuTime, errorVal;

    StopWatchInterface *hTimer = NULL;
    int i;

    sdkCreateTimer(&hTimer);

    printf("Generating input data...\n");
    //Generate options set
    srand(123);

    for (i = 0; i < OPT_N; i++)
    {
        optionData[i].S = randData(5.0f, 30.0f);
        optionData[i].X = randData(1.0f, 100.0f);
        optionData[i].T = randData(0.25f, 10.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
        BlackScholesCall(callValueBS[i], optionData[i]);
    }

    printf("Running GPU binomial tree...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    binomialOptionsGPU(callValueGPU, optionData, OPT_N);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("Options count            : %i     \n", OPT_N);
    printf("Time steps               : %i     \n", NUM_STEPS);
    printf("binomialOptionsGPU() time: %f msec\n", gpuTime);
    printf("Options per second       : %f     \n", OPT_N / (gpuTime * 0.001));

    printf("Running CPU binomial tree...\n");

    for (i = 0; i < OPT_N; i++)
    {
        binomialOptionsCPU(callValueCPU[i], optionData[i]);
    }

    printf("Comparing the results...\n");
    sumDelta = 0;
    sumRef   = 0;
    printf("GPU binomial vs. Black-Scholes\n");

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("CPU binomial vs. Black-Scholes\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i]- callValueCPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("CPU binomial vs. GPU binomial\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
        sumRef += callValueCPU[i];
    }

    if (sumRef > 1E-5)
    {
        printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("Shutting down...\n");

    sdkDeleteTimer(&hTimer);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    if (errorVal > 5e-4)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
