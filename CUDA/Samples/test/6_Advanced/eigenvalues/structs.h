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

/* Helper structures to simplify variable handling */

#ifndef _STRUCTS_H_
#define _STRUCTS_H_

struct InputData
{

    //! host side representation of diagonal
    float  *a;
    //! host side representation superdiagonal
    float  *b;

    //! device side representation of diagonal
    float  *g_a;
    //! device side representation of superdiagonal
    float  *g_b;
    //! helper variable pointing to the mem allocated for g_b which provides
    //! space for one additional element of padding at the beginning
    float  *g_b_raw;

};

struct ResultDataSmall
{

    //! eigenvalues (host side)
    float *eigenvalues;

    // left interval limits at the end of the computation
    float *g_left;

    // right interval limits at the end of the computation
    float *g_right;

    // number of eigenvalues smaller than the left interval limit
    unsigned int *g_left_count;

    // number of eigenvalues bigger than the right interval limit
    unsigned int *g_right_count;

    //! flag if algorithm converged
    unsigned int *g_converged;

    // helper variables

    unsigned int mat_size_f;
    unsigned int mat_size_ui;

    float         *zero_f;
    unsigned int  *zero_ui;
};


struct ResultDataLarge
{

    // number of intervals containing one eigenvalue after the first step
    unsigned int *g_num_one;

    // number of (thread) blocks of intervals containing multiple eigenvalues
    // after the first step
    unsigned int *g_num_blocks_mult;

    //! left interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_left_one;

    //! right interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_right_one;

    //! interval indices (position in sorted listed of eigenvalues)
    //! of intervals containing one eigenvalue after the first iteration step
    unsigned int *g_pos_one;

    //! left interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_left_mult;

    //! right interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_right_mult;

    //! number of eigenvalues less than the left limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_left_count_mult;

    //! number of eigenvalues less than the right limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_right_count_mult;

    //! start addresses in g_left_mult etc. of blocks of intervals containing
    //! more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult;

    //! accumulated number of intervals in g_left_mult etc. of blocks of
    //! intervals containing more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult_sum;

    //! eigenvalues that have been generated in the second step from intervals
    //! that still contained multiple eigenvalues after the first step
    float *g_lambda_mult;

    //! eigenvalue index of intervals that have been generated in the second
    //! processing step
    unsigned int *g_pos_mult;

};

#endif // #ifndef _STRUCTS_H_

