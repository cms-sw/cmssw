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
 * Routines for testing the device API of CUBLAS.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Includes, cuda helper functions */
#include <helper_cuda.h>

__global__ void invokeDeviceCublasSgemm(cublasStatus_t *returnValue,
                                        int n,
                                        const float *d_alpha,
                                        const float *d_A,
                                        const float *d_B,
                                        const float *d_beta,
                                        float *d_C)
{
    cublasHandle_t cnpHandle;
    cublasStatus_t status = cublasCreate(&cnpHandle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        *returnValue = status;
        return;
    }

    /* Perform operation using cublas */
    status =
        cublasSgemm(cnpHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    d_alpha,
                    d_A, n,
                    d_B, n,
                    d_beta,
                    d_C, n);

    cublasDestroy(cnpHandle);

    *returnValue = status;
}

struct SGEMMScalarParams
{
    float alpha, beta;
};

extern "C" void device_cublas_sgemm(int n,
                                    float alpha,
                                    const float *d_A, const float *d_B,
                                    float beta,
                                    float *d_C)
{
    cublasStatus_t *d_status;
    cublasStatus_t status;

    if (cudaMalloc((void **) &d_status, sizeof(cublasStatus_t)) != cudaSuccess)
    {
        fprintf(stderr,
                "!!!! device memory allocation error (allocate d_status)\n");
        exit(EXIT_FAILURE);
    }

    // Device API requires scalar arguments (alpha and beta)
    // to be allocated in the device memory.
    SGEMMScalarParams h_params = {alpha, beta};
    SGEMMScalarParams *d_params;

    if (cudaMalloc((void **) &d_params, sizeof(SGEMMScalarParams)) !=
        cudaSuccess)
    {
        fprintf(stderr,
                "!!!! device memory allocation error (allocate d_params)\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_params,
                   &h_params,
                   sizeof(SGEMMScalarParams),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr,
                "!!!! host to device memory copy error\n");
        exit(EXIT_FAILURE);
    }

    // Launch cublasSgemm wrapper kernel.
    invokeDeviceCublasSgemm<<<1, 1>>>
    (d_status, n, &d_params->alpha, d_A, d_B, &d_params->beta, d_C);

    cudaError_t error;

    if ((error = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr,
                "!!!! kernel execution error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(&status,
                   d_status,
                   sizeof(cublasStatus_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr,
                "!!!! device to host memory copy error\n");
        exit(EXIT_FAILURE);
    }

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "!!!! CUBLAS Device API call failed with code %d\n",
                status);
        exit(EXIT_FAILURE);
    }

    // Free allocated device memory.
    if (cudaFree(d_status) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_status)\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_params) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_params)\n");
        exit(EXIT_FAILURE);
    }
}
