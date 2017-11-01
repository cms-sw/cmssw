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
 * This example demonstrates how to call CUBLAS library
 * functions both from the HOST code and from the DEVICE code
 * running on the GPU (the latter is available only for the compute
 * capability >= 3.5). The single-precision matrix-matrix
 * multiplication operation, SGEMM, will be performed 3 times:
 * 1) once by calling a method defined in this file (simple_sgemm),
 * 2) once by calling the cublasSgemm library routine from the HOST code
 * 3) and once by calling the cublasSgemm library routine from
 *    the DEVICE code.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Includes, cuda helper functions */
#include <helper_cuda.h>

/* Matrix size */
#define N  (275)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Checks result against reference and returns relative error */
static float check_result(const float *result,
                          const float *reference,
                          int size)
{
    float error_norm = 0.0f;
    float ref_norm = 0.0f;

    for (int i = 0; i < size; ++i)
    {
        float diff = reference[i] - result[i];
        error_norm += diff * diff;
        ref_norm += reference[i] * reference[i];
    }

    error_norm = (float) sqrt((double) error_norm);
    ref_norm = (float) sqrt((double) ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! Result check failed: reference norm is 0\n");
        exit(EXIT_FAILURE);
    }

    return error_norm / ref_norm;
}

/* Declaration of the function that computes sgemm using CUBLAS device API */
extern "C" void device_cublas_sgemm(int n,
                                    float alpha,
                                    const float *d_A, const float *d_B,
                                    float beta,
                                    float *d_C);

/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C_rnd;
    float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    cublasHandle_t handle;

    int dev_id;
    cudaDeviceProp device_prop;

    bool do_device_api_test = false;

    float host_api_test_ratio, device_api_test_ratio;

    /* Initialize CUBLAS */
    printf("simpleDevLibCUBLAS test running...\n");

    dev_id = findCudaDevice(argc, (const char **) argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

    if ((device_prop.major << 4) + device_prop.minor >= 0x35)
    {
        printf("Host and device APIs will be tested.\n");
        do_device_api_test = true;
    }
    /*    else if ((device_prop.major << 4) + device_prop.minor >= 0x20)
        {
            printf("Host API will be tested.\n");
            do_device_api_test = false;
        }
    */
    else
    {
        fprintf(stderr, "simpleDevLibCUBLAS examples requires Compute Capability of SM 3.5 or higher\n");
        return EXIT_WAIVED;
    }

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C_rnd = (float *)malloc(n2 * sizeof(h_C_rnd[0]));

    if (h_C_rnd == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C_rnd)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(n2 * sizeof(h_C_ref[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C_rnd[i] = rand() / (float)RAND_MAX;
        h_C[i] = h_C_rnd[i];
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_C_rnd[0]), h_C_rnd, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /*
     * Performs operation using plain C code
     */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /*
     * Performs operation using cublas
     */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Check result against reference */
    host_api_test_ratio = check_result(h_C, h_C_ref, n2);

    if (do_device_api_test)
    {
        /* Reset device resident C matrix */
        status = cublasSetVector(n2, sizeof(h_C_rnd[0]), h_C_rnd, 1, d_C, 1);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "!!!! device access error (write C)\n");
            return EXIT_FAILURE;
        }

        /*
         * Performs operation using the device API of CUBLAS library
         */
        device_cublas_sgemm(N, alpha, d_A, d_B, beta, d_C);

        /* Read the result back */
        status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "!!!! device access error (read C)\n");
            return EXIT_FAILURE;
        }

        /* Check result against reference */
        device_api_test_ratio = check_result(h_C, h_C_ref, n2);
    }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_rnd);
    free(h_C_ref);

    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

    bool test_result = do_device_api_test ?
                       host_api_test_ratio < 1e-6 &&
                       device_api_test_ratio < 1e-6 :
                       host_api_test_ratio < 1e-6;

    printf("simpleDevLibCUBLAS completed, returned %s\n",
           test_result ? "OK" : "ERROR!");

    exit(test_result ? EXIT_SUCCESS : EXIT_FAILURE);
}
