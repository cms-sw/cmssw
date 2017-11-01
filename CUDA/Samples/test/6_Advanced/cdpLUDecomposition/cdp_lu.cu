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

//
//  CDP_LU.CU
//
//  Test infrastructure is here. Kernels are each included in separate files.
//

#include <stdio.h>
//#include <omp.h>
#include "cdp_lu.h"
#include "cdp_lu_utils.h"
extern __global__ void dgetrf_cdpentry(Parameters *device_params);


// Entry point for dgetrf. We allocate memories and simply call the kernel.
void dgetrf_test(Parameters *host_params, Parameters *device_params)
{


    double t_start = time_in_seconds();

    // Launch the kernel (just a device-function call in CDP terms)
    dgetrf_cdpentry<<< 1, 1 >>>(device_params);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) 
    {
        printf("Failed to launch CDP kernel (%s)\nCalling exit...\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    else 
    {
        printf("Successfully launched CDP kernel\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());

    double gpu_sec = time_in_seconds() - t_start;

    // Check our return data
    /*
    for(int b=0; b<batch; b++)
    {
        if(*(params[b].hostmem.info) != 0)
            printf("Degenerate matrix %d/%d.\n", b+1, batch);
    } */

    double flop_count = (double) host_params->flop_count;
    printf("GPU perf(dgetrf)= %.3f Gflops\n", flop_count / (1000000000.*gpu_sec));
}
