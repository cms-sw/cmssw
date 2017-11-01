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

/**
**************************************************************************
* \file BmpUtil.h
* \brief Contains basic image operations declaration.
*
* This file contains declaration of basic bitmap loading, saving,
* conversions to different representations and memory management routines.
*/

#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(push)
#endif

#pragma pack(1)

typedef char                            int8;
typedef short                           int16;
typedef int                             int32;
typedef unsigned char                   uint8;
typedef unsigned short                  uint16;
typedef unsigned int                    uint32;

/**
* \brief Bitmap file header structure
*
*  Bitmap file header structure
*/
typedef struct
{
    uint16 _bm_signature;                   //!< File signature, must be "BM"
    uint32 _bm_file_size;                   //!< File size
    uint32 _bm_reserved;                    //!< Reserved, must be zero
    uint32 _bm_bitmap_data;             //!< Bitmap data
} BMPFileHeader;


/**
* \brief Bitmap info header structure
*
*  Bitmap info header structure
*/
typedef struct
{
    uint32 _bm_info_header_size;            //!< Info header size, must be 40
    uint32 _bm_image_width;             //!< Image width
    uint32 _bm_image_height;                //!< Image height
    uint16 _bm_num_of_planes;               //!< Amount of image planes, must be 1
    uint16 _bm_color_depth;             //!< Color depth
    uint32 _bm_compressed;              //!< Image compression, must be none
    uint32 _bm_bitmap_size;             //!< Size of bitmap data
    uint32 _bm_hor_resolution;          //!< Horizontal resolution, assumed to be 0
    uint32 _bm_ver_resolution;          //!< Vertical resolution, assumed to be 0
    uint32 _bm_num_colors_used;         //!< Number of colors used, assumed to be 0
    uint32 _bm_num_important_colors;        //!< Number of important colors, assumed to be 0
} BMPInfoHeader;


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(pop)
#else
#pragma pack()
#endif


/**
* \brief Simple 2D size / region_of_interest structure
*
*  Simple 2D size / region_of_interest structure
*/
typedef struct
{
    int width;          //!< ROI width
    int height;         //!< ROI height
} ROI;


/**
*  One-byte unsigned integer type
*/
typedef unsigned char byte;


extern "C"
{
    int clamp_0_255(int x);
    float round_f(float num);
    byte *MallocPlaneByte(int width, int height, int *pStepBytes);
    short *MallocPlaneShort(int width, int height, int *pStepBytes);
    float *MallocPlaneFloat(int width, int height, int *pStepBytes);
    void CopyByte2Float(byte *ImgSrc, int StrideB, float *ImgDst, int StrideF, ROI Size);
    void CopyFloat2Byte(float *ImgSrc, int StrideF, byte *ImgDst, int StrideB, ROI Size);
    void FreePlane(void *ptr);
    void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);
    void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);
    int PreLoadBmp(char *FileName, int *Width, int *Height);
    void LoadBmpAsGray(char *FileName, int Stride, ROI ImSize, byte *Img);
    void DumpBmpAsGray(char *FileName, byte *Img, int Stride, ROI ImSize);
    void DumpBlockF(float *PlaneF, int StrideF, char *Fname);
    void DumpBlock(byte *Plane, int Stride, char *Fname);
    float CalculateMSE(byte *Img1, byte *Img2, int Stride, ROI Size);
    float CalculatePSNR(byte *Img1, byte *Img2, int Stride, ROI Size);
}
