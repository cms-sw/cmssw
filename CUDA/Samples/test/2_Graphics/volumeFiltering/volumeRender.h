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

#ifndef _VOLUMERENDER__H_
#define _VOLUMERENDER__H_

#include <cuda_runtime.h>
#include "volume.h"


extern "C" {
    void VolumeRender_init();
    void VolumeRender_deinit();

    void VolumeRender_setPreIntegrated(int state);
    void VolumeRender_setVolume(const Volume *volume);
    void VolumeRender_setTextureFilterMode(bool bLinearFilter);
    void VolumeRender_render(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                             float density, float brightness, float transferOffset, float transferScale);
    void VolumeRender_copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
};


#endif

