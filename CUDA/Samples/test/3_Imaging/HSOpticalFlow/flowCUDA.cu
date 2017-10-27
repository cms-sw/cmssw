/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "common.h"

// include kernels
#include "downscaleKernel.cuh"
#include "upscaleKernel.cuh"
#include "warpingKernel.cuh"
#include "derivativesKernel.cuh"
#include "solverKernel.cuh"
#include "addKernel.cuh"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(const float *I0,
                     const float *I1,
                     int width, int height, int stride,
                     float alpha,
                     int nLevels,
                     int nWarpIters,
                     int nSolverIters,
                     float *u,
                     float *v)
{
    printf("Computing optical flow on GPU...\n");

    // pI0 and pI1 will hold device pointers
    const float **pI0 = new const float *[nLevels];
    const float **pI1 = new const float *[nLevels];

    int *pW = new int [nLevels];
    int *pH = new int [nLevels];
    int *pS = new int [nLevels];

    // device memory pointers
    float *d_tmp;
    float *d_du0;
    float *d_dv0;
    float *d_du1;
    float *d_dv1;

    float *d_Ix;
    float *d_Iy;
    float *d_Iz;

    float *d_u;
    float *d_v;
    float *d_nu;
    float *d_nv;

    const int dataSize = stride * height * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_tmp, dataSize));
    checkCudaErrors(cudaMalloc(&d_du0, dataSize));
    checkCudaErrors(cudaMalloc(&d_dv0, dataSize));
    checkCudaErrors(cudaMalloc(&d_du1, dataSize));
    checkCudaErrors(cudaMalloc(&d_dv1, dataSize));

    checkCudaErrors(cudaMalloc(&d_Ix, dataSize));
    checkCudaErrors(cudaMalloc(&d_Iy, dataSize));
    checkCudaErrors(cudaMalloc(&d_Iz, dataSize));

    checkCudaErrors(cudaMalloc(&d_u , dataSize));
    checkCudaErrors(cudaMalloc(&d_v , dataSize));
    checkCudaErrors(cudaMalloc(&d_nu, dataSize));
    checkCudaErrors(cudaMalloc(&d_nv, dataSize));

    // prepare pyramid

    int currentLevel = nLevels - 1;
    // allocate GPU memory for input images
    checkCudaErrors(cudaMalloc(pI0 + currentLevel, dataSize));
    checkCudaErrors(cudaMalloc(pI1 + currentLevel, dataSize));

    checkCudaErrors(cudaMemcpy((void *)pI0[currentLevel], I0, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)pI1[currentLevel], I1, dataSize, cudaMemcpyHostToDevice));

    pW[currentLevel] = width;
    pH[currentLevel] = height;
    pS[currentLevel] = stride;

    for (; currentLevel > 0; --currentLevel)
    {
        int nw = pW[currentLevel] / 2;
        int nh = pH[currentLevel] / 2;
        int ns = iAlignUp(nw);

        checkCudaErrors(cudaMalloc(pI0 + currentLevel - 1, ns * nh * sizeof(float)));
        checkCudaErrors(cudaMalloc(pI1 + currentLevel - 1, ns * nh * sizeof(float)));

        Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);

        Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

        pW[currentLevel - 1] = nw;
        pH[currentLevel - 1] = nh;
        pS[currentLevel - 1] = ns;
    }

    checkCudaErrors(cudaMemset(d_u, 0, stride * height * sizeof(float)));
    checkCudaErrors(cudaMemset(d_v, 0, stride * height * sizeof(float)));

    // compute flow
    for (; currentLevel < nLevels; ++currentLevel)
    {

        for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)
        {
            checkCudaErrors(cudaMemset(d_du0, 0, dataSize));
            checkCudaErrors(cudaMemset(d_dv0, 0, dataSize));

            checkCudaErrors(cudaMemset(d_du1, 0, dataSize));
            checkCudaErrors(cudaMemset(d_dv1, 0, dataSize));

            // on current level we compute optical flow
            // between frame 0 and warped frame 1
            WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                      pS[currentLevel], d_u, d_v, d_tmp);

            ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel],
                               pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

            for (int iter = 0; iter < nSolverIters; ++iter)
            {
                SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz,
                               pW[currentLevel], pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);

                Swap(d_du0, d_du1);
                Swap(d_dv0, d_dv1);
            }

            // update u, v
            Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
            Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
        }

        if (currentLevel != nLevels - 1)
        {
            // prolongate solution
            float scaleX = (float)pW[currentLevel + 1]/(float)pW[currentLevel];

            Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleX, d_nu);

            float scaleY = (float)pH[currentLevel + 1]/(float)pH[currentLevel];

            Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleY, d_nv);

            Swap(d_u, d_nu);
            Swap(d_v, d_nv);
        }
    }

    checkCudaErrors(cudaMemcpy(u, d_u, dataSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(v, d_v, dataSize, cudaMemcpyDeviceToHost));

    // cleanup
    for (int i = 0; i < nLevels; ++i)
    {
        checkCudaErrors(cudaFree((void *)pI0[i]));
        checkCudaErrors(cudaFree((void *)pI1[i]));
    }

    delete [] pI0;
    delete [] pI1;
    delete [] pW;
    delete [] pH;
    delete [] pS;

    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaFree(d_du0));
    checkCudaErrors(cudaFree(d_dv0));
    checkCudaErrors(cudaFree(d_du1));
    checkCudaErrors(cudaFree(d_dv1));
    checkCudaErrors(cudaFree(d_Ix));
    checkCudaErrors(cudaFree(d_Iy));
    checkCudaErrors(cudaFree(d_Iz));
    checkCudaErrors(cudaFree(d_nu));
    checkCudaErrors(cudaFree(d_nv));
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_v));
}
