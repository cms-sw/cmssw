/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef DDS_H
#define DDS_H

#if !defined(MAKEFOURCC)
#    define MAKEFOURCC(ch0, ch1, ch2, ch3) \
    ( (unsigned int)(ch0)        | ((unsigned int)(ch1) << 8) | \
      ( (unsigned int)(ch2) << 16) | ((unsigned int)(ch3) << 24) )
#endif

typedef unsigned int uint;
typedef unsigned short ushort;

struct DDSPixelFormat
{
    uint size;
    uint flags;
    uint fourcc;
    uint bitcount;
    uint rmask;
    uint gmask;
    uint bmask;
    uint amask;
};

struct DDSCaps
{
    uint caps1;
    uint caps2;
    uint caps3;
    uint caps4;
};

/// DDS file header.
struct DDSHeader
{
    uint fourcc;
    uint size;
    uint flags;
    uint height;
    uint width;
    uint pitch;
    uint depth;
    uint mipmapcount;
    uint reserved[11];
    DDSPixelFormat pf;
    DDSCaps caps;
    uint notused;
};

static const uint FOURCC_DDS = MAKEFOURCC('D', 'D', 'S', ' ');
static const uint FOURCC_DXT1 = MAKEFOURCC('D', 'X', 'T', '1');
static const uint DDSD_WIDTH = 0x00000004U;
static const uint DDSD_HEIGHT = 0x00000002U;
static const uint DDSD_CAPS = 0x00000001U;
static const uint DDSD_PIXELFORMAT = 0x00001000U;
static const uint DDSCAPS_TEXTURE = 0x00001000U;
static const uint DDPF_FOURCC = 0x00000004U;
static const uint DDSD_LINEARSIZE = 0x00080000U;


#endif // DDS_H
