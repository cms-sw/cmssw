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

#ifndef NV_UTIL_NPP_IMAGE_PACKED_H
#define NV_UTIL_NPP_IMAGE_PACKED_H

#include "Image.h"
#include "Pixel.h"

namespace npp
{
    template<typename D, size_t N, class A>
    class ImagePacked: public npp::Image
    {
        public:
            typedef npp::Pixel<D, N>    tPixel;
            typedef D                   tData;
            static const size_t         gnChannels = N;
            typedef npp::Image::Size    tSize;

            ImagePacked(): aPixels_(0)
                , nPitch_(0)
            {
                ;
            }

            ImagePacked(unsigned int nWidth, unsigned int nHeight): Image(nWidth, nHeight)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
            }

            ImagePacked(unsigned int nWidth, unsigned int nHeight, bool bTight): Image(nWidth, nHeight)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_, bTight);
            }

            ImagePacked(const tSize &rSize): Image(rSize)
                , aPixels_(0)
                , nPitch_(0)
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
            }

            ImagePacked(const ImagePacked<D, N, A> &rImage): Image(rImage)
                , aPixels_(0)
                , nPitch_(rImage.pitch())
            {
                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
                A::Copy2D(aPixels_, nPitch_, rImage.pixels(), rImage.pitch(), width(), height());
            }

            virtual
            ~ImagePacked()
            {
                A::Free2D(aPixels_);
            }

            ImagePacked &
            operator= (const ImagePacked<D, N, A> &rImage)
            {
                // in case of self-assignment
                if (&rImage == this)
                {
                    return *this;
                }

                A::Free2D(aPixels_);
                aPixels_ = 0;
                nPitch_ = 0;

                // assign parent class's data fields (width, height)
                Image::operator =(rImage);

                aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
                A::Copy2D(aPixels_, nPitch_, rImage.data(), rImage.pitch(), width(), height());

                return *this;
            }

            unsigned int
            pitch()
            const
            {
                return nPitch_;
            }

            /// Get a pointer to the pixel array.
            ///     The result pointer can be offset to pixel at position (x, y) and
            /// even negative offsets are allowed.
            /// \param nX Horizontal pointer/array offset.
            /// \param nY Vertical pointer/array offset.
            /// \return Pointer to the pixel array (or first pixel in array with coordinates (nX, nY).
            tPixel *
            pixels(int nX = 0, int nY = 0)
            {
                return reinterpret_cast<tPixel *>(reinterpret_cast<unsigned char *>(aPixels_) + nY * pitch() + nX * gnChannels * sizeof(D));
            }

            const
            tPixel *
            pixels(int nX = 0, int nY = 0)
            const
            {
                return reinterpret_cast<const tPixel *>(reinterpret_cast<unsigned char *>(aPixels_) + nY * pitch() + nX * gnChannels * sizeof(D));
            }

            D *
            data(int nX = 0, int nY = 0)
            {
                return reinterpret_cast<D *>(pixels(nX, nY));
            }

            const
            D *
            data(int nX = 0, int nY = 0)
            const
            {
                return reinterpret_cast<const D *>(pixels(nX, nY));
            }

            void
            swap(ImagePacked<D, N, A> &rImage)
            {
                Image::swap(rImage);

                tData *aTemp   = aPixels_;
                aPixels_        = rImage.aPixels_;
                rImage.aPixels_ = aTemp;

                unsigned int nTemp = nPitch_;
                nPitch_            = rImage.nPitch_;
                rImage.nPitch_     = nTemp;
            }

        private:
            D *aPixels_;
            unsigned int nPitch_;
    };

} // npp namespace


#endif // NV_IMAGE_IPP_H
