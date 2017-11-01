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

#ifndef _CDP_LU_UTILS_H_
#define _CDP_LU_UTILS_H_

#ifndef HELPER_CUDA_H
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __check_cuda_errors (err, __FILE__, __LINE__)

inline void __check_cuda_errors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fflush(stdout);
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString(err));

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(-1);
    }
}
#endif

#ifndef MAX
#define MAX(a, b) ((a >= b) ? a : b)
#endif
#ifndef MIN
#define MIN(a, b) ((a <= b) ? a : b)
#endif

inline void errorExit(const char *msg)
{
#ifdef SHR_QATEST_H
    shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
#else
    fflush(stdout);
    fprintf(stderr, "%s\n", msg);
    exit(-1);
#endif
}

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static __inline__ double time_in_seconds(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }

    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux) || defined(__APPLE__)
#include <sys/time.h>
static double time_in_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif


#endif /* !defined _CDP_LU_UTILS_H_ */

