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

#ifndef ASIANOPTION_H
#define ASIANOPTION_H

template <typename Real>
struct AsianOption
{
    enum CallPut {Call, Put};

    // Parameters
    Real spot;
    Real strike;
    Real r;
    Real sigma;
    Real tenor;
    Real dt;

    // Value
    Real golden;
    Real value;

    // Option type
    CallPut type;
};

#endif
