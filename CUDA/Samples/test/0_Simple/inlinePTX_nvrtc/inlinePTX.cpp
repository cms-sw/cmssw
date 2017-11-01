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
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <nvrtc_helper.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID=0; elemID<length; elemID++)
    {
        h_ptr[elemID] = elemID % 32;
    }
}

int main(int argc, char **argv)
{
    printf("CUDA inline PTX assembler sample\n");

    char *ptx, *kernel_file;
    size_t ptxSize;

    kernel_file = sdkFindFilePath("inlinePTX_kernel.cu", argv[0]);
    compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0);

    CUmodule module = loadPTX(ptx, argc, argv);

    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "sequence_gpu"));

    const int N = 1000;
    int *h_ptr = (int *)malloc(N *sizeof(int));

    dim3 cudaBlockSize(256,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);

    CUdeviceptr d_ptr;
    checkCudaErrors(cuMemAlloc(&d_ptr, N * sizeof(int)));

    void *arr[] = { (void *)&d_ptr, (void *)&N };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                                             0,0, /* shared mem, stream */
                                            &arr[0], /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());

    sequence_cpu(h_ptr, N);

    int *h_d_ptr = (int *)malloc(N * sizeof(int));
    checkCudaErrors(cuMemcpyDtoH(h_d_ptr, d_ptr, N *sizeof(int)));

    bool bValid = true;

    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
        }
    }

    printf("Test %s.\n", bValid ? "Successful" : "Failed");

    checkCudaErrors(cuMemFree(d_ptr));

    return bValid ? EXIT_SUCCESS: EXIT_FAILURE;
}
