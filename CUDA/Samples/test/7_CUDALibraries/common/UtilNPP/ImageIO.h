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

#ifndef NV_UTIL_NPP_IMAGE_IO_H
#define NV_UTIL_NPP_IMAGE_IO_H

#include "ImagesCPU.h"
#include "ImagesNPP.h"

#include "FreeImage.h"
#include "Exceptions.h"

#include <string>
#include "string.h"


// Error handler for FreeImage library.
//  In case this handler is invoked, it throws an NPP exception.
void
FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    throw npp::Exception(zMessage);
}

namespace npp
{
    // Load a gray-scale image from disk.
    void
    loadImage(const std::string &rFileName, ImageCPU_8u_C1 &rImage)
    {
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

        // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
        {
            eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
        }

        NPP_ASSERT(eFormat != FIF_UNKNOWN);
        // check that the plugin has reading capabilities ...
        FIBITMAP *pBitmap;

        if (FreeImage_FIFSupportsReading(eFormat))
        {
            pBitmap = FreeImage_Load(eFormat, rFileName.c_str());
        }

        NPP_ASSERT(pBitmap != 0);
        // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);

        // create an ImageCPU to receive the loaded image data
        ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

        // Copy the FreeImage data into the new ImageCPU
        unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
        const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) -1);
        Npp8u *pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();

        for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, oImage.width() * sizeof(Npp8u));
            pSrcLine -= nSrcPitch;
            pDstLine += nDstPitch;
        }

        // swap the user given image with our result image, effecively
        // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }

    // Save an gray-scale image to disk.
    void
    saveImage(const std::string &rFileName, const ImageCPU_8u_C1 &rImage)
    {
        // create the result image storage using FreeImage so we can easily
        // save
        FIBITMAP *pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nDstPitch   = FreeImage_GetPitch(pResultBitmap);
        Npp8u *pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height()-1);
        const Npp8u *pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();

        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, rImage.width() * sizeof(Npp8u));
            pSrcLine += nSrcPitch;
            pDstLine -= nDstPitch;
        }

        // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, rFileName.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }

    // Load a gray-scale image from disk.
    void
    loadImage(const std::string &rFileName, ImageNPP_8u_C1 &rImage)
    {
        ImageCPU_8u_C1 oImage;
        loadImage(rFileName, oImage);
        ImageNPP_8u_C1 oResult(oImage);
        rImage.swap(oResult);
    }

    // Save an gray-scale image to disk.
    void
    saveImage(const std::string &rFileName, const ImageNPP_8u_C1 &rImage)
    {
        ImageCPU_8u_C1 oHostImage(rImage.size());
        // copy the device result data
        rImage.copyTo(oHostImage.data(), oHostImage.pitch());
        saveImage(rFileName, oHostImage);
    }
}


#endif // NV_UTIL_NPP_IMAGE_IO_H
