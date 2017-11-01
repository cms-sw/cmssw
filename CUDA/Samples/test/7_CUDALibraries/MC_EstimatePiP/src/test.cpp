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

#include "../inc/test.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../inc/piestimator.h"

template <typename Real>
bool Test<Real>::operator()()
{
    using std::stringstream;
    using std::endl;
    using std::setw;

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // Get device properties
    struct cudaDeviceProp deviceProperties;
    cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, device);

    if (cudaResult != cudaSuccess)
    {
        std::string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Evaluate on GPU
    printf("Estimating Pi on GPU (%s)\n\n", deviceProperties.name);
    PiEstimator<Real> estimator(numSims, device, threadBlockSize, seed);
    sdkStartTimer(&timer);
    Real result = estimator();
    sdkStopTimer(&timer);
    elapsedTime = sdkGetAverageTimerValue(&timer)/1000.0f;

    // Tolerance to compare result with expected
    // This is just to check that nothing has gone very wrong with the
    // test, the actual accuracy of the result depends on the number of
    // Monte Carlo trials
    const Real tolerance = static_cast<Real>(0.01);

    // Display results
    Real abserror = fabs(result - static_cast<float>(PI));
    Real relerror = abserror / static_cast<float>(PI);
    printf("Precision:      %s\n", (typeid(Real) == typeid(double)) ? "double" : "single");
    printf("Number of sims: %d\n", numSims);
    printf("Tolerance:      %e\n", tolerance);
    printf("GPU result:     %e\n", result);
    printf("Expected:       %e\n", PI);
    printf("Absolute error: %e\n", abserror);
    printf("Relative error: %e\n\n", relerror);

    // Check result
    if (relerror > tolerance)
    {
        printf("computed result (%e) does not match expected result (%e).\n", result, PI);
        pass = false;
    }
    else
    {
        pass = true;
    }

    // Print results
    printf("MonteCarloEstimatePiP, Performance = %.2f sims/s, Time = %.2f(ms), NumDevsUsed = %u, Blocksize = %u\n",
           numSims / elapsedTime, elapsedTime*1000.0f, 1, threadBlockSize);

    return pass;
}

// Explicit template instantiation
template struct Test<float>;
template struct Test<double>;
