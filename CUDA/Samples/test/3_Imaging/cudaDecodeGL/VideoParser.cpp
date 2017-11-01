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

#include "VideoParser.h"

#include "VideoDecoder.h"
#include "FrameQueue.h"

#include <cstring>
#include <cassert>

VideoParser::VideoParser(VideoDecoder *pVideoDecoder, FrameQueue *pFrameQueue, CUVIDEOFORMATEX *pFormat, CUcontext *pCudaContext) : hParser_(0)
{
    assert(0 != pFrameQueue);
    oParserData_.pFrameQueue   = pFrameQueue;
    assert(0 != pVideoDecoder);
    oParserData_.pVideoDecoder = pVideoDecoder;
    oParserData_.pContext      = pCudaContext;

    CUVIDPARSERPARAMS oVideoParserParameters;
    memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
    oVideoParserParameters.CodecType              = pVideoDecoder->codec();
    oVideoParserParameters.ulMaxNumDecodeSurfaces = pVideoDecoder->maxDecodeSurfaces();
    oVideoParserParameters.ulMaxDisplayDelay      = 1;  // this flag is needed so the parser will push frames out to the decoder as quickly as it can
    oVideoParserParameters.pUserData              = &oParserData_;
    oVideoParserParameters.pExtVideoInfo          = pFormat;
    oVideoParserParameters.pfnSequenceCallback    = HandleVideoSequence;    // Called before decoding frames and/or whenever there is a format change
    oVideoParserParameters.pfnDecodePicture       = HandlePictureDecode;    // Called when a picture is ready to be decoded (decode order)
    oVideoParserParameters.pfnDisplayPicture      = HandlePictureDisplay;   // Called whenever a picture is ready to be displayed (display order)
    CUresult oResult = cuvidCreateVideoParser(&hParser_, &oVideoParserParameters);
    assert(CUDA_SUCCESS == oResult);
}

int
CUDAAPI
VideoParser::HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat)
{
    VideoParserData *pParserData = reinterpret_cast<VideoParserData *>(pUserData);

    if ((pFormat->codec != cudaVideoCodec_VP9) && ((pFormat->coded_width != pParserData->pVideoDecoder->frameWidth())
        || (pFormat->coded_height != pParserData->pVideoDecoder->frameHeight())))
    {
        // Only VP9 supports dynamic resolution Change.
        return 0;
    }
    if ((pFormat->codec != pParserData->pVideoDecoder->codec())         // codec-type
        || (pFormat->chroma_format != pParserData->pVideoDecoder->chromaFormat()))
    {
        // We don't deal with dynamic changes in video format
        return 0;
    }

    return 1;
}

int
CUDAAPI
VideoParser::HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams)
{
    VideoParserData *pParserData = reinterpret_cast<VideoParserData *>(pUserData);

    bool bFrameAvailable = pParserData->pFrameQueue->waitUntilFrameAvailable(pPicParams->CurrPicIdx);

    if (!bFrameAvailable)
        return false;

    if (pParserData->pVideoDecoder->decodePicture(pPicParams, pParserData->pContext) != CUDA_SUCCESS)
    {
        return false;
    }

    return true;
}

int
CUDAAPI
VideoParser::HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pPicParams)
{
    // std::cout << *pPicParams << std::endl;

    VideoParserData *pParserData = reinterpret_cast<VideoParserData *>(pUserData);

    pParserData->pFrameQueue->enqueue(pPicParams);

    return 1;
}

std::ostream &
operator << (std::ostream &rOutputStream, const CUVIDPARSERDISPINFO &rParserDisplayInfo)
{
    rOutputStream << "Picture Index: " << rParserDisplayInfo.picture_index << "\n";
    rOutputStream << "Progressive frame: ";

    if (rParserDisplayInfo.progressive_frame)
        rOutputStream << "true\n";
    else
        rOutputStream << "false\n";

    rOutputStream << "Top field first: ";

    if (rParserDisplayInfo.top_field_first)
        rOutputStream << "true\n";
    else
        rOutputStream << "false\n";

    rOutputStream << "Repeat first field: ";

    if (rParserDisplayInfo.repeat_first_field)
        rOutputStream << "true\n";
    else
        rOutputStream << "false\n";

    rOutputStream << "Time stamp: " << rParserDisplayInfo.timestamp << "\n";

    return rOutputStream;
}

