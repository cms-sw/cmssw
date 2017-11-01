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
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

const char *sSDKname     = "conjugateGradientUM";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

int main(int argc, char **argv)
{
    int N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    const int max_iter = 10000;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    float dot;
    float *r, *p, *Ax;
    int k;
    float alpha, beta, alpham1;

    printf("Starting [%s]...\n", sSDKname);

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (!deviceProp.managedMemory) { 
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    N = 1048576;
    nz = (N-2)*3 + 4;

    cudaMallocManaged((void **)&I, sizeof(int)*(N+1));
    cudaMallocManaged((void **)&J, sizeof(int)*nz);
    cudaMallocManaged((void **)&val, sizeof(float)*nz);

    genTridiag(I, J, val, N, nz);

    cudaMallocManaged((void **)&x, sizeof(float)*N);
    cudaMallocManaged((void **)&rhs, sizeof(float)*N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    // temp memory for CG
    checkCudaErrors(cudaMallocManaged((void **)&r, N*sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&p, N*sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&Ax, N*sizeof(float)));

    cudaDeviceSynchronize();

    for (int i=0; i < N; i++)
    {
        r[i] = rhs[i];
    }

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, x, &beta, Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, r, 1, p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, p, &beta, Ax);
        cublasStatus = cublasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    printf("Final residual: %e\n",sqrt(r1));

    fprintf(stdout,"&&&& conjugateGradientUM %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    cudaFree(I);
    cudaFree(J);
    cudaFree(val);
    cudaFree(x);
    cudaFree(rhs);
    cudaFree(r);
    cudaFree(p);
    cudaFree(Ax);

    printf("Test Summary:  Error amount = %f, result = %s\n", err, (k <= max_iter) ? "SUCCESS" : "FAILURE");
    exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
}
