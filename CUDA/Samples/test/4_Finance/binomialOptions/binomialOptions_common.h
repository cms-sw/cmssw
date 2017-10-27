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



#ifndef BINOMIALOPTIONS_COMMON_H
#define BINOMIALOPTIONS_COMMON_H

#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct
{
    real S;
    real X;
    real T;
    real R;
    real V;
} TOptionData;


////////////////////////////////////////////////////////////////////////////////
// Global parameters
////////////////////////////////////////////////////////////////////////////////
//Number of time steps
#define   NUM_STEPS 2048
//Max option batch size
#define MAX_OPTIONS 1024



#endif
