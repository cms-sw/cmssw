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

#ifndef SCAN_COMMON_H
#define SCAN_COMMON_H

#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
extern "C" const uint MAX_BATCH_ELEMENTS;
extern "C" const uint MIN_SHORT_ARRAY_SIZE;
extern "C" const uint MAX_SHORT_ARRAY_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE;

////////////////////////////////////////////////////////////////////////////////
// CUDA scan
////////////////////////////////////////////////////////////////////////////////
extern "C" void initScan(void);
extern "C" void closeScan(void);

extern "C" size_t scanExclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
);

extern "C" size_t scanExclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
);

////////////////////////////////////////////////////////////////////////////////
// Reference CPU scan
////////////////////////////////////////////////////////////////////////////////
extern "C" void scanExclusiveHost(
    uint *dst,
    uint *src,
    uint batchSize,
    uint arrayLength
);

#endif
