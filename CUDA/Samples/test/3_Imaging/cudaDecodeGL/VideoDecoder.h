/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_VIDEODECODER_H
#define NV_VIDEODECODER_H

#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cuviddec.h" // <cuviddec.h>
#include "dynlink_nvcuvid.h"  // <nvcuvid.h>

#define MAX_FRAME_COUNT 2

// Wrapper class around the CUDA Video Decoding API.
//
class VideoDecoder
{
    public:
        explicit
        VideoDecoder(const CUVIDEOFORMAT &rVideoFormat, CUcontext &rContext,
                     cudaVideoCreateFlags eCreateFlags, CUvideoctxlock &ctx);

        ~VideoDecoder();

        // Get the code-type currently used.
        cudaVideoCodec
        codec()
        const;

        cudaVideoChromaFormat
        chromaFormat()
        const;

        // Maximum number of decode surfaces used by decoder.
        unsigned long
        maxDecodeSurfaces()
        const;

        unsigned long
        frameWidth()
        const;

        unsigned long
        frameHeight()
        const;

        unsigned long
        targetWidth()
        const;

        unsigned long
        targetHeight()
        const;

        CUresult
        decodePicture(CUVIDPICPARAMS *pPictureParameters, CUcontext *pContext = NULL);

        CUresult
        mapFrame(int iPictureIndex, CUdeviceptr *ppDevice, unsigned int *nPitch, CUVIDPROCPARAMS *pVideoProcessingParameters);

        CUresult
        unmapFrame(CUdeviceptr pDevice);

    private:
        // Default constructor. Don't implement.
        VideoDecoder();

        // Copy constructor. Don't implement.
        VideoDecoder(const VideoDecoder &);

        // Assignment operator. Don't implement.
        void
        operator= (const VideoDecoder &);

        CUVIDDECODECREATEINFO   oVideoDecodeCreateInfo_;
        CUvideodecoder          oDecoder_;
        cudaVideoCreateFlags    m_VideoCreateFlags;
        CUcontext               m_Context;
        CUvideoctxlock          m_VidCtxLock;
};

#endif // NV_VIDEODECODER_H
