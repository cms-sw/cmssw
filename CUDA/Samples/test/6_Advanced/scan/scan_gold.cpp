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


#include "scan_common.h"



extern "C" void scanExclusiveHost(
    uint *dst,
    uint *src,
    uint batchSize,
    uint arrayLength
)
{
    for (uint i = 0; i < batchSize; i++, src += arrayLength, dst += arrayLength)
    {
        dst[0] = 0;

        for (uint j = 1; j < arrayLength; j++)
            dst[j] = src[j - 1] + dst[j - 1];
    }
}
