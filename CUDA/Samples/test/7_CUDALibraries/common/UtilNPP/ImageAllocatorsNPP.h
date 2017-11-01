/*
* Copyright 2008-2009 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#ifndef NV_UTIL_NPP_IMAGE_ALLOCATORS_NPP_H
#define NV_UTIL_NPP_IMAGE_ALLOCATORS_NPP_H

#include "Exceptions.h"

#include <nppi.h>
#include <cuda_runtime.h>

namespace npp
{
    template <typename D, size_t N>
    D *
    MallocTightCUDA(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch)
    {
        D *pResult;
        *pPitch = nWidth * sizeof(D) * N;
        NPP_CHECK_CUDA(cudaMalloc(&pResult, *pPitch * nHeight));
        NPP_ASSERT_NOT_NULL(pResult);

        return pResult;
    }


    template <typename D, size_t N>
    class ImageAllocator
    {
    };

    template<>
    class ImageAllocator<Npp8u, 1>
    {
        public:
            static
            Npp8u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp8u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp8u, 1>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_8u_C1(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp8u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp8u, 2>
    {
        public:
            static
            Npp8u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp8u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp8u, 2>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_8u_C2(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp8u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp8u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp8u, 3>
    {
        public:
            static
            Npp8u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp8u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp8u, 3>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_8u_C3(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp8u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp8u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp8u, 4>
    {
        public:
            static
            Npp8u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp8u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp8u, 4>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_8u_C4(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp8u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp8u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16u, 1>
    {
        public:
            static
            Npp16u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16u, 1>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16u_C1(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16u, 2>
    {
        public:
            static
            Npp16u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16u, 2>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16u_C2(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };


    template<>
    class ImageAllocator<Npp16u, 3>
    {
        public:
            static
            Npp16u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16u, 3>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16u_C3(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp16u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16u, 4>
    {
        public:
            static
            Npp16u *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16u *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16u, 4>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16u_C4(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16u *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16u), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16u *pDst, size_t nDstPitch, const Npp16u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16u), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16s, 1>
    {
        public:
            static
            Npp16s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16s, 1>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16s_C1(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16s, 2>
    {
        public:
            static
            Npp16s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16s, 2>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16s_C2(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp16s, 4>
    {
        public:
            static
            Npp16s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp16s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp16s, 4>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_16s_C4(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp16s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp16s *pDst, size_t nDstPitch, const Npp16s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp16s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32s, 1>
    {
        public:
            static
            Npp32s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32s, 1>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32s_C1(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32s, 3>
    {
        public:
            static
            Npp32s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32s, 3>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32s_C3(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32s, 4>
    {
        public:
            static
            Npp32s *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32s *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32s, 4>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32s_C4(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32s *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32s), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32s *pDst, size_t nDstPitch, const Npp32s *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32s), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32f, 1>
    {
        public:
            static
            Npp32f *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32f *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32f, 1>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32f_C1(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32f *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32f, 2>
    {
        public:
            static
            Npp32f *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32f *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32f, 2>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32f_C2(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32f *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp32f), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 2 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32f, 3>
    {
        public:
            static
            Npp32f *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32f *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32f, 3>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32f_C3(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32f *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32f), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 3 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

    template<>
    class ImageAllocator<Npp32f, 4>
    {
        public:
            static
            Npp32f *
            Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int *pPitch, bool bTight = false)
            {
                NPP_ASSERT(nWidth * nHeight > 0);

                Npp32f *pResult = 0;

                if (bTight)
                {
                    pResult = MallocTightCUDA<Npp32f, 4>(nWidth, nHeight, pPitch);
                }
                else
                {
                    pResult = nppiMalloc_32f_C4(nWidth, nHeight, reinterpret_cast<int *>(pPitch));
                    NPP_ASSERT(pResult != 0);
                }

                return pResult;
            };

            static
            void
            Free2D(Npp32f *pPixels)
            {
                nppiFree(pPixels);
            };

            static
            void
            Copy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32f), nHeight, cudaMemcpyHostToDevice);
                NPP_ASSERT(cudaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy2D(Npp32f *pDst, size_t nDstPitch, const Npp32f *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
            {
                cudaError_t eResult;
                eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * 4 * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToHost);
                NPP_ASSERT(cudaSuccess == eResult);
            };
    };

} // npp namespace

#endif // NV_UTIL_NPP_IMAGE_ALLOCATORS_NPP_H
