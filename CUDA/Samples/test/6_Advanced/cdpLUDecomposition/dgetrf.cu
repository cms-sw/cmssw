/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is being provided
* under the terms and conditions of a Source Code License Agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <cublas_v2.h>
#include "cdp_lu.h"

extern __device__ void report_error(const char *strName, int info);
extern __device__ __noinline__ void dgetf2(cublasHandle_t cb_handle, int m, int n, double *A, int lda, int *ipiv, int *info, cg::thread_block cta);
extern __global__ void dlaswp(int n, double *A, int lda, int *ipiv, int k1, int k2);

#define DGETRF_BLOCK_SIZE 32

__device__ __noinline__ void dgetrf(cublasHandle_t cb_handle, cudaStream_t stream, int m, int n, double *A, int lda, int *ipiv, int *info, cg::thread_block cta)
{
    cublasStatus_t status;

    // The flag set by one thread to indicate a failure.
    __shared__ int s_info;

    // Initialize to 0
    if (threadIdx.x == 0)
    {
        s_info = 0;
    }

    *info = 0;

    if (m < 0)
    {
        *info = -1;
    }

    if (n < 0)
    {
        *info = -2;
    }

    if (lda < max(1, m))
    {
        *info = -4;
    }

    if (*info)
    {
        if (threadIdx.x == 0)
            report_error("DGETRF", *info);

        return;
    }

    // Quick return if possible

    if (m == 0 || n == 0)
    {
        return;
    }

    // Determine the block size for this environment.

    int nb = 64;
    const int minDim = min(m, n);

    if (nb < 1 || nb > minDim)
    {
        // We're too small - fall through to just calling dgetf2.
        dgetf2(cb_handle, m, n, A, lda, ipiv, info, cta);
        return;
    }

    // Big enough to use blocked code.
    for (int j = 0 ; j < minDim ; j += nb)
    {
        int iinfo;
        int jb = min(minDim - j, nb);

        if (threadIdx.x == 0)
        {
            // Factor diagonal and subdiagonal blocks and test for exact singularity.
            dgetf2(cb_handle, m-j, jb, &A[j*lda + j], lda, &ipiv[j], &iinfo, cta);

            // Adjust INFO and the pivot indices.
            if (*info == 0 && iinfo > 0)
                s_info = iinfo + j;
        }

        cg::sync(cta);

        // Make sure info is valid.
        *info = s_info;

        // We update ipiv in parallel on the device, if we were launched with >1 threads
        for (int i = j+threadIdx.x, end = min(m, j+jb) ; i < end ; i += blockDim.x)
            ipiv[i] += j;

        cg::sync(cta);

        // Apply interchanges to columns 1:J-1. JB rows.
        if (threadIdx.x == 0)
        {
            if (j > 0)
                dlaswp<<<1, 256, 0, stream>>>(j, A, lda, ipiv, j, j+jb);

            // Apply interchanges to columns J+JB:N. JB rows.
            if (j+jb < n)
            {
                dlaswp<<<1, 256, 0, stream>>>(n-j-jb, &A[(j+jb)*lda], lda, ipiv, j, j+jb);

                double one = 1.0;
                status = cublasDtrsm_v2(
                             cb_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             jb, n-j-jb,
                             &one,
                             &A[j*lda+j], lda,
                             &A[(j+jb)*lda+j], lda);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    printf("dgetrf: Failed dtrsm: %d\n", status);
                    s_info = 1;
                }
            }
        }

        cg::sync(cta);

        // Make sure info has the correct value.
        if (s_info)
        {
            *info = s_info;
            return;
        }

        // Update trailing submatrix.
        if (threadIdx.x == 0 && j + jb < m)
        {
            double one = 1.0;
            double minus_one = -1.0;
            status = cublasDgemm_v2(
                         cb_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         m-j-jb, n-j-jb, jb,
                         &minus_one,
                         &A[j*lda + j+jb], lda,
                         &A[(j+jb)*lda + j], lda,
                         &one,
                         &A[(j+jb)*lda + j+jb], lda);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                printf("dgetrf: Failed dgemm: %d\n", status);
                s_info = 1;
            }
        }

        cg::sync(cta);

        // Make sure info has the correct value.
        if (s_info)
        {
            *info = s_info;
            return;
        }
    }
}

////////////////////////////////////////////////////////////
//
//  Entry functions for host-side and device-side calling
//
////////////////////////////////////////////////////////////

__global__ void dgetrf_cdpentry(Parameters *device_params)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cublasHandle_t cb_handle = NULL;

    cudaStream_t stream;

    if (threadIdx.x == 0)
    {
        cublasStatus_t status = cublasCreate_v2(&cb_handle);
        cublasSetPointerMode_v2(cb_handle, CUBLAS_POINTER_MODE_HOST);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            *device_params->device_info = -8;
            printf("dgetrf: Failed to create cublas context - status = %d\n", status);
            return;
        }

        // Create a local stream for all of our operations
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasSetStream_v2(cb_handle, stream);
    }

    cg::sync(cta);        // Compiler requires this to not tail-split the if()...

    dgetrf(cb_handle,
           stream,
           device_params->m,
           device_params->n,
           device_params->device_LU,
           device_params->lda,
           device_params->device_piv,
           device_params->device_info, cta);
}
