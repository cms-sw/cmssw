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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
#include <nvrtc_helper.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


const char *sampleName = "simpleAtomicIntrinsics_nvrtc";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    printf("%s completed, returned %s\n", sampleName, testResult ? "OK" : "ERROR!");

    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char **argv)
{
    int dev = 0;

    char *ptx, *kernel_file;
    size_t ptxSize;

    kernel_file = sdkFindFilePath("simpleAtomicIntrinsics_kernel.cuh", argv[0]);
    compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0);

    CUmodule module = loadPTX(ptx, argc, argv);
    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "testKernel"));

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData = (int *) malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    // allocate device memory for result
    CUdeviceptr dOData;
    checkCudaErrors(cuMemAlloc(&dOData, memSize));
    checkCudaErrors(cuMemcpyHtoD(dOData, hOData, memSize));

    // execute the kernel
    dim3 cudaBlockSize(numThreads,1,1);
    dim3 cudaGridSize(numBlocks, 1, 1);


    void *arr[] = { (void *)&dOData };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                                            0,0, /* shared mem, stream */
                                            &arr[0], /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());

    checkCudaErrors(cuMemcpyDtoH(hOData, dOData, memSize));

    //Copy result from device to host
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Compute reference solution
    testResult = computeGold(hOData, numThreads * numBlocks);

    // Cleanup memory
    free(hOData);
    checkCudaErrors(cuMemFree(dOData));
}
