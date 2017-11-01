/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_UTIL_NPP_IMAGE_ALLOCATORS_CPU_H
#define NV_UTIL_NPP_IMAGE_ALLOCATORS_CPU_H

#include "Exceptions.h"

namespace npp
{

    template <typename D, size_t N>
    class ImageAllocatorCPU
    {
        public:
            static
            D *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                D *pResult = new D[nWidth * N * nHeight];
                *pPitch = nWidth * sizeof(D) * N;

                return pResult;
            };

            static
            void
            Free2D(D *pPixels)
            {
                delete[] pPixels;
            };

            static
            void
            Copy2D(D *pDst, size_t nDstPitch, const D *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                const void *pSrcLine = pSrc;
                void        *pDstLine = pDst;

                for (size_t iLine = 0; iLine < nHeight; ++iLine)
                {
                    // copy one line worth of data
                    memcpy(pDst, pSrc, nWidth * N * sizeof(D));
                    // move data pointers to next line
                    pDst += nDstPitch;
                    pSrc += nSrcPitch;
                }
            };

    };

} // npp namespace

#endif // NV_UTIL_NPP_IMAGE_ALLOCATORS_CPU_H
