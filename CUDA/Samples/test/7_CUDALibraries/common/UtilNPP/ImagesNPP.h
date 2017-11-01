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

#ifndef NV_UTIL_NPP_IMAGES_NPP_H
#define NV_UTIL_NPP_IMAGES_NPP_H

#include "Exceptions.h"
#include "ImagePacked.h"

#include "ImageAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
    // forward declaration
    template<typename D, unsigned int N, class A> class ImageCPU;

    template<typename D, unsigned int N>
    class ImageNPP: public npp::ImagePacked<D, N, npp::ImageAllocator<D, N> >
    {
        public:
            ImageNPP()
            {
                ;
            }

            ImageNPP(unsigned int nWidth, unsigned int nHeight, bool bTight = false): ImagePacked<D, N, npp::ImageAllocator<D, N> >(nWidth, nHeight, bTight)
            {
                ;
            }

            ImageNPP(const npp::Image::Size &rSize): ImagePacked<D, N, npp::ImageAllocator<D, N> >(rSize)
            {
                ;
            }

            ImageNPP(const ImageNPP<D, N> &rImage): Image(rImage)
            {
                ;
            }

            template<class X>
            explicit
            ImageNPP(const ImageCPU<D, N, X> &rImage, bool bTight = false): ImagePacked<D, N, npp::ImageAllocator<D, N> >(rImage.width(), rImage.height(), bTight)
            {
                npp::ImageAllocator<D, N>::HostToDeviceCopy2D(ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(),
                                                              rImage.data(),
                                                              rImage.pitch(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
            }

            virtual
            ~ImageNPP()
            {
                ;
            }

            ImageNPP &
            operator= (const ImageNPP<D, N> &rImage)
            {
                ImagePacked<D, N, npp::ImageAllocator<D, N> >::operator= (rImage);

                return *this;
            }

            void
            copyTo(D *pData, unsigned int nPitch)
            const
            {
                NPP_ASSERT((ImagePacked<D, N, npp::ImageAllocator<D, N> >::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
                npp::ImageAllocator<D, N>::DeviceToHostCopy2D(pData,
                                                              nPitch,
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
            }

            void
            copyFrom(D *pData, unsigned int nPitch)
            {
                NPP_ASSERT((ImagePacked<D, N, npp::ImageAllocator<D, N> >::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
                npp::ImageAllocator<D, N>::HostToDeviceCopy2D(ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(),
                                                              pData,
                                                              nPitch,
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(),
                                                              ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
            }
    };

    typedef ImageNPP<Npp8u,  1>   ImageNPP_8u_C1;
    typedef ImageNPP<Npp8u,  2>   ImageNPP_8u_C2;
    typedef ImageNPP<Npp8u,  3>   ImageNPP_8u_C3;
    typedef ImageNPP<Npp8u,  4>   ImageNPP_8u_C4;

    typedef ImageNPP<Npp16u, 1>  ImageNPP_16u_C1;
    typedef ImageNPP<Npp16u, 2>  ImageNPP_16u_C2;
    typedef ImageNPP<Npp16u, 3>  ImageNPP_16u_C3;
    typedef ImageNPP<Npp16u, 4>  ImageNPP_16u_C4;

    typedef ImageNPP<Npp16s, 1>  ImageNPP_16s_C1;
    typedef ImageNPP<Npp16s, 3>  ImageNPP_16s_C3;
    typedef ImageNPP<Npp16s, 4>  ImageNPP_16s_C4;

    typedef ImageNPP<Npp32s, 1>  ImageNPP_32s_C1;
    typedef ImageNPP<Npp32s, 3>  ImageNPP_32s_C3;
    typedef ImageNPP<Npp32s, 4>  ImageNPP_32s_C4;

    typedef ImageNPP<Npp32f, 1>  ImageNPP_32f_C1;
    typedef ImageNPP<Npp32f, 2>  ImageNPP_32f_C2;
    typedef ImageNPP<Npp32f, 3>  ImageNPP_32f_C3;
    typedef ImageNPP<Npp32f, 4>  ImageNPP_32f_C4;

    typedef ImageNPP<Npp64f, 1>  ImageNPP_64f_C1;
    typedef ImageNPP<Npp64f, 2>  ImageNPP_64f_C2;
    typedef ImageNPP<Npp64f, 3>  ImageNPP_64f_C3;
    typedef ImageNPP<Npp64f, 4>  ImageNPP_64f_C4;

} // npp namespace

#endif // NV_UTIL_NPP_IMAGES_NPP_H
