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

#ifndef NV_VIDEO_PARSER
#define NV_VIDEO_PARSER

#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cuviddec.h" // <cuviddec.h>
#include "dynlink_nvcuvid.h"  // <nvcuvid.h>

#include <iostream>

class FrameQueue;
class VideoDecoder;

// Wrapper class around the CUDA video-parser API.
//  The video parser consumes a video-data stream and parses it into
// a) Sequences: Whenever a new sequence or initial sequence header is found
//      in the video stream, the parser calls its sequence-handling callback
//      function.
// b) Decode segments: Whenever a a completed frame or half-frame is found
//      the parser calls its picture decode callback.
// c) Display: Whenever a complete frame was decoded, the parser calls the
//      display picture callback.
//
class VideoParser
{
    public:
        // Constructor.
        //
        // Parameters:
        //      pVideoDecoder - pointer to valid VideoDecoder object. This VideoDecoder
        //          is used in the parser-callbacks to decode video-frames.
        //      pFrameQueue - pointer to a valid FrameQueue object. The FrameQueue is used
        //          by  the parser-callbacks to store decoded frames in it.
        VideoParser(VideoDecoder *pVideoDecoder, FrameQueue *pFrameQueue, CUVIDEOFORMATEX *pFormat, CUcontext *pCudaContext = NULL);

    private:
        // Struct containing user-data to be passed by parser-callbacks.
        struct VideoParserData
        {
            VideoDecoder *pVideoDecoder;
            FrameQueue    *pFrameQueue;
            CUcontext     *pContext;
        };

        // Default constructor. Don't implement.
        explicit
        VideoParser();

        // Copy constructor. Don't implement.
        VideoParser(const VideoParser &);

        // Assignment operator. Don't implement.
        void
        operator= (const VideoParser &);

        // Called when the decoder encounters a video format change (or initial sequence header)
        // This particular implementation of the callback returns 0 in case the video format changes
        // to something different than the original format. Returning 0 causes a stop of the app.
        static
        int
        CUDAAPI
        HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat);

        // Called by the video parser to decode a single picture
        // Since the parser will deliver data as fast as it can, we need to make sure that the picture
        // index we're attempting to use for decode is no longer used for display
        static
        int
        CUDAAPI
        HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams);

        // Called by the video parser to display a video frame (in the case of field pictures, there may be
        // 2 decode calls per 1 display call, since two fields make up one frame)
        static
        int
        CUDAAPI
        HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pPicParams);


        VideoParserData oParserData_;   // instance of the user-data we have passed into the parser-callbacks.
        CUvideoparser   hParser_;       // handle to the CUDA video-parser

        friend class VideoSource;
};

std::ostream &
operator << (std::ostream &rOutputStream, const CUVIDPARSERDISPINFO &rParserDisplayInfo);

#endif // NV_VIDEO_PARSER

