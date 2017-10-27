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

#ifndef IMAGE_DENOISING_H
#define IMAGE_DENOISING_H

typedef unsigned int TColor;

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS   3
#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3
#define KNN_WINDOW_AREA     ( (2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1) )
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define INV_KNN_WINDOW_AREA ( 1.0f / (float)KNN_WINDOW_AREA )
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

#define KNN_WEIGHT_THRESHOLD    0.02f
#define KNN_LERP_THRESHOLD      0.79f
#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

// functions to load images
extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH);
extern "C" void cuda_KNN(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);
extern "C" void cuda_KNNdiag(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);
extern "C" void cuda_NLM(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);
extern "C" void cuda_NLMdiag(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);

extern "C" void cuda_NLM2(TColor *d_dst, int imageW, int imageH, float Noise, float LerpC);
extern "C" void cuda_NLM2diag(TColor *d_dst, int imageW, int imageH, float Noise, float LerpC);

#endif
