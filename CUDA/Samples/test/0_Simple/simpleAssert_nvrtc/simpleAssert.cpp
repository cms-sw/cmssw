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

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/utsname.h>
#endif

// Includes, system
#include <stdio.h>
#include <cassert>

// Includes CUDA
#include <cuda_runtime.h>
#include "nvrtc_helper.h"

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

const char *sampleName = "simpleAssert_nvrtc";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;


////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}



void runTest(int argc, char **argv)
{
    int Nblocks = 2;
    int Nthreads = 32;

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.

    dim3 dimGrid(Nblocks);
    dim3 dimBlock(Nthreads);

    printf("Launch kernel to generate assertion failures\n");
    char *ptx, *kernel_file;
    size_t ptxSize;

    kernel_file = sdkFindFilePath("simpleAssert_kernel.cu", argv[0]);
    compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0);

    CUmodule module = loadPTX(ptx, argc, argv);
    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "testKernel"));

    int count = 60;
    void *args[] = { (void *)&count };

    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            dimGrid.x, dimGrid.y, dimGrid.z, /* grid dim */
                                            dimBlock.x, dimBlock.y, dimBlock.z, /* block dim */
                                            0,0, /* shared mem, stream */
                                            &args[0], /* arguments */
                                            0));

    //Synchronize (flushes assert output).
    printf("\n-- Begin assert output\n\n");
    CUresult res = cuCtxSynchronize();

    printf("\n-- End assert output\n\n");

    //Check for errors and failed asserts in asynchronous kernel launch.
    if (res == CUDA_ERROR_ASSERT)
    {
        printf("Device assert failed as expected\n");
    }

    testResult = res == CUDA_ERROR_ASSERT ;
}

