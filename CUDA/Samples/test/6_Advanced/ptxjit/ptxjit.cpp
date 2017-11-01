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

/*
 * This sample uses the Driver API to just-in-time compile (JIT) a Kernel from PTX code.
 * Additionally, this sample demonstrates the seamless interoperability capability of CUDA runtime
 * Runtime and CUDA Driver API calls.
 * This sample requires Compute Capability 2.0 and higher.
 *
 */

// System includes
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>

// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// sample include
#include "ptxjit.h"

const char *sSDKname = "PTX Just In Time (JIT) Compilation (no-qatest)";


void ptxJIT(int argc, char **argv, CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState)
{
    CUjit_option options[6];
    void *optionVals[6];
    float walltime;
    char error_log[8192],
         info_log[8192];
    unsigned int logSize = 8192;
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *) (long)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *) (long) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6,options, optionVals, lState));

    if (sizeof(void *)==4)
    {
        // Load the PTX from the string myPtx32
        printf("Loading myPtx32[] program\n");
        // PTX May also be loaded from file, as per below.
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)myPtx32, strlen(myPtx32)+1, 0, 0, 0, 0);
    }
    else
    {
        // Load the PTX from the string myPtx (64-bit)
        printf("Loading myPtx[] program\n");
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)myPtx64, strlen(myPtx64)+1, 0, 0, 0, 0);
        // PTX May also be loaded from file, as per below.
        // myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "myPtx64.ptx",0,0,0);
    }

    if (myErr != CUDA_SUCCESS)
    {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n",walltime,info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(phModule, cuOut));

    // Locate the kernel entry poin
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "_Z8myKernelPi"));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
}

int main(int argc, char **argv)
{
    const unsigned int nThreads = 256;
    const unsigned int nBlocks  = 64;
    const size_t memSize = nThreads * nBlocks * sizeof(int);

    CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    CUlinkState  lState;
    int         *d_data   = 0;
    int         *h_data   = 0;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;

    printf("[%s] - Starting...\n", sSDKname);

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        cuda_device = getCmdLineArgumentInt(argc, (const char **)argv, "device=");

        if (cuda_device < 0)
        {
            printf("Invalid command line parameters\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            printf("cuda_device = %d\n", cuda_device);
            cuda_device = gpuDeviceInit(cuda_device);

            if (cuda_device < 0)
            {
                printf("No CUDA Capable devices found, exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with the highest Gflops/s
        cuda_device = gpuGetMaxGflopsDeviceId();
    }

    checkCudaErrors(cudaSetDevice(cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    printf("> Using CUDA device [%d]: %s\n", cuda_device, deviceProp.name);

    // Allocate memory on host and device (Runtime API)
    // NOTE: The runtime API will create the GPU Context implicitly here
    if ((h_data = (int *)malloc(memSize)) == NULL)
    {
        std::cerr << "Could not allocate host memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(&d_data, memSize));

    // JIT Compile the Kernel from PTX and get the Handles (Driver API)
    ptxJIT(argc, argv, &hModule, &hKernel, &lState);

    // Set the kernel parameters (Driver API)
    checkCudaErrors(cuFuncSetBlockShape(hKernel, nThreads, 1, 1));
    int paramOffset = 0;
    checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_data, sizeof(d_data)));
    paramOffset += sizeof(d_data);
    checkCudaErrors(cuParamSetSize(hKernel, paramOffset));

    // Launch the kernel (Driver API_)
    checkCudaErrors(cuLaunchGrid(hKernel, nBlocks, 1));
    std::cout << "CUDA kernel launched" << std::endl;

    // Copy the result back to the host
    checkCudaErrors(cudaMemcpy(h_data, d_data, memSize, cudaMemcpyDeviceToHost));

    // Check the result
    bool dataGood = true;

    for (unsigned int i = 0 ; dataGood && i < nBlocks * nThreads ; i++)
    {
        if (h_data[i] != (int)i)
        {
            std::cerr << "Error at " << i << std::endl;
            dataGood = false;
        }
    }

    // Cleanup
    if (d_data)
    {
        checkCudaErrors(cudaFree(d_data));
        d_data = 0;
    }

    if (h_data)
    {
        free(h_data);
        h_data = 0;
    }

    if (hModule)
    {
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }

    return dataGood ? EXIT_SUCCESS : EXIT_FAILURE;
}
