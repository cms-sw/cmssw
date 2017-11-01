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

#ifndef TEST_H
#define TEST_H

template <typename Real>
struct Test
{
    Test() : pass(false) {};

    int          device;
    unsigned int numSims;
    unsigned int threadBlockSize;
    unsigned int seed;

    bool   pass;
    double elapsedTime;

    bool operator()();
};

// Defaults are arbitrary to give sensible runtime
#define k_sims_min    100000
#define k_sims_max    1000000
#define k_sims_def    100000
#define k_sims_qa     100000
#define k_bsize_min   32
#define k_bsize_def   128
#define k_bsize_qa    128
#define k_seed_def    1234
#define k_seed_qa     1234

#endif
