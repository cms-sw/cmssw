/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The user has to prepare a sparse matrix of "matrix market format" (with extension .mtx).
 *  For example, the user can download matrices in Florida Sparse Matrix Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by the switch -R<solver> and
 *  to provide the path of the matrix by the switch -F<file>, then
 *  the program solves
 *          A*x = b  where b = ones(m,1)
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|)
 *
 *  The elapsed time is also reported so the user can compare efficiency of different solvers.
 *
 *  The runtime of linear solver contains symbolic analysis, numerical factorization and solve.
 *  The user can set environmental variable OMP_NUM_THREADS to configure number of cores in CPU path.
 *
 *  How to use
        /cuSolverSp_LinearSolver            // Default: Cholesky, symrcm & file=lap2D_5pt_n100.mtx
 *     ./cuSolverSp_LinearSolver -R=chol  -file=<file>   // cholesky factorization
 *     ./cuSolverSp_LinearSolver -R=lu -P=symrcm -file=<file>     // symrcm + LU with partial pivoting
 *     ./cuSolverSp_LinearSolver -R=qr -P=symamd -file=<file>     // symamd + QR factorization
 *
 *
 *  Remark: the absolute error on solution x is meaningless without knowing condition number of A.
 *     The relative error on residual should be close to machine zero, i.e. 1.e-15.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

#include "helper_cuda.h"
#include "helper_cusolver.h"


template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

void UsageSP(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-R=<name>   : choose a linear solver\n");
    printf( "              chol (cholesky factorization), this is default\n");
    printf( "              qr   (QR factorization)\n");
    printf( "              lu   (LU factorization)\n");
    printf( "-P=<name>    : choose a reordering\n");
    printf( "              symrcm (Reverse Cuthill-McKee)\n");
    printf( "              symamd (Approximate Minimum Degree)\n");
    printf( "-file=<filename> : filename containing a matrix in MM format\n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}


void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageSP();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "R"))
    {
        char *solverType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

        if (solverType)
        {
            if ((STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0) && (STRCASECMP(solverType, "qr") != 0))
            {
                printf("\nIncorrect argument passed to -R option\n");
                UsageSP();
            }
            else
            {
                opts.testFunc = solverType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "P"))
    {
        char *reorderType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

        if (reorderType)
        {
            if ((STRCASECMP(reorderType, "symrcm") != 0) && (STRCASECMP(reorderType, "symamd") != 0))
            {
                printf("\nIncorrect argument passed to -P option\n");
                UsageSP();
            }
            else
            {
                opts.reorder = reorderType;
            }
        }
    }

    if (!opts.reorder)
    {
        opts.reorder = "symrcm"; // Setting default reordering to be symrcm.
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageSP();
        }
    }
}

int main (int argc, char *argv[])
{
    struct testOpts opts;
    cusolverSpHandle_t handle = NULL;
    cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL;

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format

    // CSR(A) from I/O
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    double *h_csrValA = NULL;

    double *h_x = NULL; // x = A \ b
    double *h_b = NULL; // b = ones(m,1)
    double *h_r = NULL; // r = b - A*x

    int *h_Q = NULL; // <int> n
                     // reorder to reduce zero fill-in
                     // Q = symrcm(A) or Q = symamd(A)
    // B = Q*A*Q^T
    int *h_csrRowPtrB = NULL; // <int> n+1
    int *h_csrColIndB = NULL; // <int> nnzA
    double *h_csrValB = NULL; // <double> nnzA
    int *h_mapBfromA = NULL;  // <int> nnzA

    size_t size_perm = 0;
    void *buffer_cpu = NULL; // working space for permutation: B = Q*A*Q^T

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_x = NULL; // x = A \ b
    double *d_b = NULL; // a copy of h_b
    double *d_r = NULL; // r = b - A*x

    double tol = 1.e-12;
    int reorder = 0; // no reordering
    int singularity = 0; // -1 if A is invertible under tol.

    // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;

    double x_inf = 0.0;
    double r_inf = 0.0;
    double A_inf = 0.0;
    int errors = 0;
    int issym = 0;

    double start, stop;
    double time_solve_cpu;
    double time_solve_gpu;

    parseCommandLineArguments(argc, argv, opts);

    if (NULL == opts.testFunc)
    {
        opts.testFunc = "chol"; // By default running Cholesky as NO solver selected with -R option.
    }

    findCudaDevice(argc, (const char **)argv);

    if (opts.sparse_mat_filename == NULL)
    {
        opts.sparse_mat_filename =  sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find lap2D_5pt_n100.mtx\n");
    }
    else
    {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }

    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename == NULL)
    {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return EXIT_FAILURE;
    }

    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
           &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true))
    {
        exit(EXIT_FAILURE);
    }
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}
    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    if ( rowsA != colsA ){
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    checkCudaErrors(cusolverSpCreate(&handle));
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusolverSpSetStream(handle, stream));

    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

    checkCudaErrors(cusparseCreateMatDescr(&descrA));

    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));

    if (baseA)
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    h_x = (double*)malloc(sizeof(double)*colsA);
    h_b = (double*)malloc(sizeof(double)*rowsA);
    h_r = (double*)malloc(sizeof(double)*rowsA);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(double)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double)*rowsA));

    // verify if A has symmetric pattern or not
    checkCudaErrors(cusolverSpXcsrissymHost(
        handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrRowPtrA+1, h_csrColIndA, &issym));

    if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        if (!issym)
        {
            printf("Error: A has no symmetric pattern, please use LU or QR \n");
            exit(EXIT_FAILURE);
        }
    }

    if (NULL != opts.reorder)
    {
        printf("step 2: reorder the matrix A to minimize zero fill-in\n");
        printf("        if the user choose a reordering by -P=symrcm or -P=symamd\n");
        printf("        The reordering will overwrite A such that \n");
        printf("            A := A(Q,Q) where Q = symrcm(A) or Q = symamd(A)\n");

        h_Q          = (int*   )malloc(sizeof(int)*colsA);
        h_csrRowPtrB = (int*   )malloc(sizeof(int)*(rowsA+1));
        h_csrColIndB = (int*   )malloc(sizeof(int)*nnzA);
        h_csrValB    = (double*)malloc(sizeof(double)*nnzA);
        h_mapBfromA  = (int*   )malloc(sizeof(int)*nnzA);

        assert(NULL != h_Q);
        assert(NULL != h_csrRowPtrB);
        assert(NULL != h_csrColIndB);
        assert(NULL != h_csrValB   );
        assert(NULL != h_mapBfromA);

        if ( 0 == strcmp(opts.reorder, "symrcm") )
        {
            checkCudaErrors(cusolverSpXcsrsymrcmHost(
                handle, rowsA, nnzA,
                descrA, h_csrRowPtrA, h_csrColIndA,
                h_Q));
        }
        else if ( 0 == strcmp(opts.reorder, "symamd") )
        {
            checkCudaErrors(cusolverSpXcsrsymamdHost(
                handle, rowsA, nnzA,
                descrA, h_csrRowPtrA, h_csrColIndA,
                h_Q));
        }
        else 
        {
            fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
            return 1;
        }

        // B = Q*A*Q^T
        memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int)*(rowsA+1));
        memcpy(h_csrColIndB, h_csrColIndA, sizeof(int)*nnzA);

        checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
            handle, rowsA, colsA, nnzA,
            descrA, h_csrRowPtrB, h_csrColIndB,
            h_Q, h_Q,
            &size_perm));

        if (buffer_cpu)
        {
            free(buffer_cpu);
        }
        buffer_cpu = (void*)malloc(sizeof(char)*size_perm);
        assert(NULL != buffer_cpu);

        // h_mapBfromA = Identity
        for(int j = 0 ; j < nnzA ; j++)
        {
            h_mapBfromA[j] = j;
        }
        checkCudaErrors(cusolverSpXcsrpermHost(
            handle, rowsA, colsA, nnzA,
            descrA, h_csrRowPtrB, h_csrColIndB,
            h_Q, h_Q,
            h_mapBfromA,
            buffer_cpu));

        // B = A( mapBfromA )
        for(int j = 0 ; j < nnzA ; j++)
        {
            h_csrValB[j] = h_csrValA[ h_mapBfromA[j] ];
        }

        // A := B
        memcpy(h_csrRowPtrA, h_csrRowPtrB, sizeof(int)*(rowsA+1));
        memcpy(h_csrColIndA, h_csrColIndB, sizeof(int)*nnzA);
        memcpy(h_csrValA   , h_csrValB   , sizeof(double)*nnzA);

        printf("step 2.1: set right hand side vector (b) to 1\n");
    }
    else
    {
        printf("step 2: set right hand side vector (b) to 1\n");
    }

    for(int row = 0 ; row < rowsA ; row++)
    {
        h_b[row] = 1.0;
    }

    printf("step 3: prepare data on device\n");
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));

    printf("step 4: solve A*x = b on CPU\n");
    // A and b are read-only
    start = second();
    start = second();

    if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        checkCudaErrors(cusolverSpDcsrlsvcholHost(
            handle, rowsA, nnzA,
            descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
            h_b, tol, reorder, h_x, &singularity));
    }
    else if ( 0 == strcmp(opts.testFunc, "lu") )
    { 
        checkCudaErrors(cusolverSpDcsrlsvluHost(
            handle, rowsA, nnzA,
            descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
            h_b, tol, reorder, h_x, &singularity));

    }
    else if ( 0 == strcmp(opts.testFunc, "qr") )
    { 
        checkCudaErrors(cusolverSpDcsrlsvqrHost(
            handle, rowsA, nnzA,
            descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
            h_b, tol, reorder, h_x, &singularity));

    }
    else
    {
        fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
        return 1;
    }
    stop = second();

    time_solve_cpu = stop - start;

    if (0 <= singularity)
    {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
    }

    printf("step 5: evaluate residual r = b - A*x (result on CPU)\n");
    checkCudaErrors(cudaMemcpy(d_r, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, sizeof(double)*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cusparseDcsrmv(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        rowsA, 
        colsA,
        nnzA,
        &minus_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA, 
        d_x,
        &one,
        d_r));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

    printf("(CPU) |b - A*x| = %E \n", r_inf);
    printf("(CPU) |A| = %E \n", A_inf);
    printf("(CPU) |x| = %E \n", x_inf);
    printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

    printf("step 6: solve A*x = b on GPU\n");
    // d_A and d_b are read-only
    start = second();
    start = second();

    if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        checkCudaErrors(cusolverSpDcsrlsvchol(
            handle, rowsA, nnzA,
            descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
            d_b, tol, reorder, d_x, &singularity));

    }
    else if ( 0 == strcmp(opts.testFunc, "lu") )
    {
        printf("WARNING: no LU available on GPU \n");
    }
    else if ( 0 == strcmp(opts.testFunc, "qr") )
    {
        checkCudaErrors(cusolverSpDcsrlsvqr(
            handle, rowsA, nnzA,
            descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
            d_b, tol, reorder, d_x, &singularity));
    }
    else
    {
        fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
        return 1;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve_gpu = stop - start;

    if (0 <= singularity)
    {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
    }

    printf("step 7: evaluate residual r = b - A*x (result on GPU)\n");
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusparseDcsrmv(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        rowsA,
        colsA,
        nnzA,
        &minus_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_x,
        &one,
        d_r));

    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);

    if ( 0 != strcmp(opts.testFunc, "lu") )
    {
        // only cholesky and qr have GPU version
        printf("(GPU) |b - A*x| = %E \n", r_inf);
        printf("(GPU) |A| = %E \n", A_inf);
        printf("(GPU) |x| = %E \n", x_inf);
        printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));
    }

    fprintf (stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n", opts.testFunc, time_solve_cpu, time_solve_gpu);

    if (handle) { checkCudaErrors(cusolverSpDestroy(handle)); }
    if (cusparseHandle) { checkCudaErrors(cusparseDestroy(cusparseHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }
    if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }

    if (h_csrValA   ) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_r) { free(h_r); }

    if (h_Q) { free(h_Q); }

    if (h_csrRowPtrB) { free(h_csrRowPtrB); }
    if (h_csrColIndB) { free(h_csrColIndB); }
    if (h_csrValB   ) { free(h_csrValB   ); }
    if (h_mapBfromA ) { free(h_mapBfromA ); }

    if (buffer_cpu) { free(buffer_cpu); }

    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }

    return 0;
}

