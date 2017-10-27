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

#include "VideoSource.h"

#include "FrameQueue.h"
#include "VideoParser.h"

#include <assert.h>

VideoSource::VideoSource(const std::string sFileName, FrameQueue *pFrameQueue): hVideoSource_(0)
{
    // fill in SourceData struct as much as we can
    // right now. Client must specify parser at a later point
    // to avoid crashes (see setParser() method).
    assert(0 != pFrameQueue);
    oSourceData_.hVideoParser = 0;
    oSourceData_.pFrameQueue = pFrameQueue;

    CUVIDSOURCEPARAMS oVideoSourceParameters;
    // Fill parameter struct
    memset(&oVideoSourceParameters, 0, sizeof(CUVIDSOURCEPARAMS));
    oVideoSourceParameters.pUserData = &oSourceData_;               // will be passed to data handlers
    oVideoSourceParameters.pfnVideoDataHandler = HandleVideoData;   // our local video-handler callback
    oVideoSourceParameters.pfnAudioDataHandler = 0;
    // now create the actual source
    CUresult oResult = cuvidCreateVideoSource(&hVideoSource_, sFileName.c_str(), &oVideoSourceParameters);
    assert(CUDA_SUCCESS == oResult);
}

VideoSource::~VideoSource()
{
    cuvidDestroyVideoSource(hVideoSource_);
}

void
VideoSource::ReloadVideo(const std::string sFileName, FrameQueue *pFrameQueue, VideoParser *pVideoParser)
{
    // fill in SourceData struct as much as we can right now. Client must specify parser at a later point
    assert(0 != pFrameQueue);
    oSourceData_.hVideoParser = pVideoParser->hParser_;
    oSourceData_.pFrameQueue  = pFrameQueue;

    cuvidDestroyVideoSource(hVideoSource_);

    CUVIDSOURCEPARAMS oVideoSourceParameters;
    // Fill parameter struct
    memset(&oVideoSourceParameters, 0, sizeof(CUVIDSOURCEPARAMS));
    oVideoSourceParameters.pUserData = &oSourceData_;               // will be passed to data handlers
    oVideoSourceParameters.pfnVideoDataHandler = HandleVideoData;   // our local video-handler callback
    oVideoSourceParameters.pfnAudioDataHandler = 0;
    // now create the actual source
    CUresult oResult = cuvidCreateVideoSource(&hVideoSource_, sFileName.c_str(), &oVideoSourceParameters);
    assert(CUDA_SUCCESS == oResult);
}


CUVIDEOFORMAT
VideoSource::format()
const
{
    CUVIDEOFORMAT oFormat;
    CUresult oResult = cuvidGetSourceVideoFormat(hVideoSource_, &oFormat, 0);
    assert(CUDA_SUCCESS == oResult);

    return oFormat;
}

void
VideoSource::getSourceDimensions(unsigned int &width, unsigned int &height)
{
    CUVIDEOFORMAT rCudaVideoFormat=  format();

    width  = rCudaVideoFormat.coded_width;
    height = rCudaVideoFormat.coded_height;
}

void
VideoSource::getDisplayDimensions(unsigned int &width, unsigned int &height)
{
    CUVIDEOFORMAT rCudaVideoFormat=  format();

    width  = abs(rCudaVideoFormat.display_area.right  - rCudaVideoFormat.display_area.left);
    height = abs(rCudaVideoFormat.display_area.bottom - rCudaVideoFormat.display_area.top);
}

void
VideoSource::getProgressive(bool &progressive)
{
    CUVIDEOFORMAT rCudaVideoFormat=  format();
    progressive = (rCudaVideoFormat.progressive_sequence != 0);
}

void
VideoSource::setParser(VideoParser &rVideoParser)
{
    oSourceData_.hVideoParser = rVideoParser.hParser_;
}

void
VideoSource::start()
{
    CUresult oResult = cuvidSetVideoSourceState(hVideoSource_, cudaVideoState_Started);
    assert(CUDA_SUCCESS == oResult);
}

void
VideoSource::stop()
{
    CUresult oResult = cuvidSetVideoSourceState(hVideoSource_, cudaVideoState_Stopped);
    assert(CUDA_SUCCESS == oResult);
}

bool
VideoSource::isStarted()
{
    return (cuvidGetVideoSourceState(hVideoSource_) == cudaVideoState_Started);
}

int
VideoSource::HandleVideoData(void *pUserData, CUVIDSOURCEDATAPACKET *pPacket)
{
    VideoSourceData *pVideoSourceData = (VideoSourceData *)pUserData;
    // Parser calls back for decode & display within cuvidParseVideoData
    CUresult oResult = cuvidParseVideoData(pVideoSourceData->hVideoParser, pPacket);

    if ((pPacket->flags & CUVID_PKT_ENDOFSTREAM) || (oResult != CUDA_SUCCESS))
        pVideoSourceData->pFrameQueue->endDecode();

    return !pVideoSourceData->pFrameQueue->isEndOfDecode();
}

std::ostream &
operator << (std::ostream &rOutputStream, const CUVIDEOFORMAT &rCudaVideoFormat)
{
    rOutputStream << "\tVideoCodec      : ";

    if ((rCudaVideoFormat.codec <= cudaVideoCodec_UYVY) &&
        (rCudaVideoFormat.codec >= cudaVideoCodec_MPEG1) &&
        (rCudaVideoFormat.codec != cudaVideoCodec_NumCodecs))
    {
        rOutputStream << eVideoFormats[rCudaVideoFormat.codec].name << "\n";
    }
    else
    {
        rOutputStream << "unknown\n";
    }

    rOutputStream << "\tFrame rate      : " << rCudaVideoFormat.frame_rate.numerator << "/" << rCudaVideoFormat.frame_rate.denominator;
    rOutputStream << "fps ~ " << rCudaVideoFormat.frame_rate.numerator/static_cast<float>(rCudaVideoFormat.frame_rate.denominator) << "fps\n";
    rOutputStream << "\tSequence format : ";

    if (rCudaVideoFormat.progressive_sequence)
        rOutputStream << "Progressive\n";
    else
        rOutputStream << "Interlaced\n";

    rOutputStream << "\tCoded frame size: [" << rCudaVideoFormat.coded_width << ", " << rCudaVideoFormat.coded_height << "]\n";
    rOutputStream << "\tDisplay area    : [" << rCudaVideoFormat.display_area.left << ", " << rCudaVideoFormat.display_area.top;
    rOutputStream << ", " << rCudaVideoFormat.display_area.right << ", " << rCudaVideoFormat.display_area.bottom << "]\n";
    rOutputStream << "\tChroma format   : ";

    switch (rCudaVideoFormat.chroma_format)
    {
        case cudaVideoChromaFormat_Monochrome:
            rOutputStream << "Monochrome\n";
            break;

        case cudaVideoChromaFormat_420:
            rOutputStream << "4:2:0\n";
            break;

        case cudaVideoChromaFormat_422:
            rOutputStream << "4:2:2\n";
            break;

        case cudaVideoChromaFormat_444:
            rOutputStream << "4:4:4\n";
            break;

        default:
            rOutputStream << "unknown\n";
    }

    rOutputStream << "\tBitrate         : ";

    if (rCudaVideoFormat.bitrate == 0)
        rOutputStream << "unknown\n";
    else
        rOutputStream << rCudaVideoFormat.bitrate/1024 << "kBit/s\n";

    rOutputStream << "\tAspect ratio    : " << rCudaVideoFormat.display_aspect_ratio.x << ":" << rCudaVideoFormat.display_aspect_ratio.y << "\n";

    return rOutputStream;
}

