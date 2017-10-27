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

#ifndef _VOLUMEFILTER_KERNEL_H_
#define _VOLUMEFILTER_KERNEL_H_

#define VOLUMEFILTER_MAXWEIGHTS 125

#include <cuda_runtime.h>
#include "volume.h"

extern "C" {
    Volume *VolumeFilter_runFilter(Volume *input, Volume *output0, Volume *output1,
                                   int iterations, int numWeights, float4 *weights, float postWeightOffset);
};


#endif

