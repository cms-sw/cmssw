/*
*  -- LAPACK routine (version 3.2) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
INTEGER            INFO, LDA, M, N
*     ..
*     .. Array Arguments ..
INTEGER            IPIV( * )
DOUBLE PRECISION   A( LDA, * )
*     ..
*
*  Purpose
*  =======
*
*  DGETF2 computes an LU factorization of a general m-by-n matrix A
*  using partial pivoting with row interchanges.
*
*  The factorization has the form
*     A = P * L * U
*  where P is a permutation matrix, L is lower triangular with unit
*  diagonal elements (lower trapezoidal if m > n), and U is upper
*  triangular (upper trapezoidal if m < n).
*
*  This is the right-looking Level 2 BLAS version of the algorithm.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the m by n matrix to be factored.
*          On exit, the factors L and U from the factorization
*          A = P*L*U; the unit diagonal elements of L are not stored.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  IPIV    (output) INTEGER array, dimension (min(M,N))
*          The pivot indices; for 1 <= i <= min(M,N), row i of the
*          matrix was interchanged with row IPIV(i).
*
*  INFO    (output) INTEGER
*          = 0: successful exit
*          < 0: if INFO = -k, the k-th argument had an illegal value
*          > 0: if INFO = k, U(k,k) is exactly zero. The factorization
*               has been completed, but the factor U is exactly
*               singular, and division by zero will occur if it is used
*               to solve a system of equations.
*
*/

#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <cublas_v2.h>


/* Input error reporting function, C version */
__device__ void report_error(const char *strName, int info)
{
    printf(" ** On entry to %s parameter number %d had an illegal value\n", strName, info);
}

__device__ __noinline__ void dgetf2(cublasHandle_t cb_handle, int m, int n, double *A, int lda, int *ipiv, int *info, cg::thread_block cta)
{
    cublasStatus_t status;

    // The flag set by one thread to indicate a failure.
    __shared__ int s_info;

    // Initialize to 0.
    if (threadIdx.x == 0)
        s_info = 0;

    cg::sync(cta);

    // Basic     argument checking
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
        report_error("DGETF2", *info);
        return;
    }

    // Quick return if possible
    if (m == 0 || n == 0)
    {
        return;
    }

    // Compute machine safe minimum
    const int minDim = min(m, n);

    // Set up the pivot array, unless it was passed in to us already set-up
    for (int j=0; j < minDim; j++)
    {
        int jp = 0;

        if (threadIdx.x == 0)
        {
            status = cublasIdamax_v2(cb_handle, m-j, &A[j*lda + j], 1, &jp);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                printf("Failed idamax: %d\n", status);
                s_info = 1;
            }

            jp += j-1; // cublasIdamax_v2 is 1-indexed (so remove 1).
            ipiv[j] = jp;
        }

        cg::sync(cta);

        // Make sure both s_info and s_jp are valid.
        if (s_info)
        {
            *info = s_info;
            return;
        }

        // Load the value A(jp, j).
        double rowval = threadIdx.x == 0 ? A[j*lda + jp] : 0.0;

        // Only threadIdx.x == 0, can be different from 0.0.
        if (threadIdx.x == 0 && rowval != 0.0 && jp != j)
        {
            status = cublasDswap_v2(cb_handle, n, &A[j], lda, &A[jp], lda);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                printf("Failed dswap: %d\n", status);
                s_info = 1;
            }
        }

        cg::sync(cta);

        // Make sure s_info has the correct value.
        if (s_info)
        {
            *info = s_info;
            return;
        }

        // Compute elements J+1:M of J-th column.
        if (threadIdx.x == 0 && rowval != 0.0 && j < m)
        {
            double scale = 1.0 / rowval;
            status = cublasDscal_v2(cb_handle, m-j-1, &scale, &A[j*lda + j+1], 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                printf("Failed dscal: %d\n", status);
                s_info = 1;
            }
        }
        else if (threadIdx.x == 0 && rowval != 0.0)
        {
            s_info = j;
        }

        cg::sync(cta);

        // Make sure s_info has the correct value.
        if (s_info)
        {
            *info = s_info;
            return;
        }

        if (threadIdx.x == 0 && j < minDim)
        {
            // Update trailing submatrix.
            double alpha = -1.0;
            status = cublasDger_v2(cb_handle, m-j-1, n-j-1, &alpha, &A[j*lda + j+1], 1, &A[(j+1)*lda + j], lda, &A[(j+1)*lda + j+1], lda);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                printf("Failed dger: %d\n", status);
                s_info = 1;
            }
        }

        cg::sync(cta);

        // Make sure s_info has the correct value.
        if (s_info)
        {
            *info = s_info;
            return;
        }
    }
}
