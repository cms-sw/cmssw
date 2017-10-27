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



#ifndef SORTINGNETWORKS_COMMON_CUH
#define SORTINGNETWORKS_COMMON_CUH



#include "sortingNetworks_common.h"

//Enables maximum occupancy
#define SHARED_SIZE_LIMIT 1024U

//Map to single instructions on G8x / G9x / G100
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )



__device__ inline void Comparator(
    uint &keyA,
    uint &valA,
    uint &keyB,
    uint &valB,
    uint dir
)
{
    uint t;

    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}



#endif
