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

/* This example demonstrates how to use the Video Decode Library with CUDA
 * bindings to interop between NVDECODE(using CUDA surfaces) and OpenGL (PBOs).
 * Post-Processing video (de-interlacing) is supported with this sample.
 */

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA Header includes
#include "dynlink_nvcuvid.h" // <nvcuvid.h>
#include "dynlink_cuda.h"    // <cuda.h>
#include "dynlink_cudaGL.h"  // <cudaGL.h>
#include "dynlink_builtin_types.h"

// CUDA utilities and system includes
#include "helper_functions.h"
#include "helper_cuda_drvapi.h"

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>

// cudaDecodeGL related helper functions
#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageGL.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

#if !defined (WIN32) && !defined (_WIN32) && !defined(WIN64) && !defined(_WIN64)
typedef unsigned char BYTE;
#define S_OK true;
#endif

const char *sAppName     = "CUDA/OpenGL Video Decode";
const char *sAppFilename = "cudaDecodeGL";
const char *sSDKname     = "cudaDecodeGL";

#define VIDEO_SOURCE_FILE "plush1_720p_10s.m2v"

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    0
#else
#define ENABLE_DEBUG_OUT    0
#endif

StopWatchInterface *frame_timer = NULL,
                    *global_timer = NULL;

int                 g_DeviceID    = 0;
bool                g_bWindowed   = true;
bool                g_bDeviceLost = false;
bool                g_bDone       = false;
bool                g_bRunning    = false;
bool                g_bAutoQuit   = false;
bool                g_bUseVsync   = false;
bool                g_bFrameRepeat= false;
bool                g_bFrameStep  = false;
bool                g_bQAReadback = false;
bool                g_bGLVerify   = false;
bool                g_bFirstFrame = true;
bool                g_bLoop       = false;
bool                g_bUpdateCSC  = true;
bool                g_bUpdateAll  = false;
bool                g_bLinearFiltering = false;
bool                g_bUseDisplay = true; // this flag enables/disables video on the window
bool                g_bUseInterop = true;
bool                g_bReadback   = false; // this flag enables/disables reading back of a video from a window
bool                g_bWriteFile  = false; // this flag enables/disables writing of a file
bool                g_bPSNR = false; // if this flag is set true, then we want to compute the PSNR
bool                g_bIsProgressive = true; // assume it is progressive, unless otherwise noted
bool                g_bException  = false;
bool                g_bWaived     = false;

int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats
long                g_nFrameStart   = -1;
long                g_nFrameEnd     = -1;

int   *pArgc = NULL;
char **pArgv = NULL;

FILE *fpWriteYUV = NULL;
FILE *fpRefYUV = NULL;

cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
CUvideoctxlock       g_CtxLock = NULL;

float present_fps, decoded_fps, total_time = 0.0f;

// These are CUDA function pointers to the CUDA kernels
CUmoduleManager   *g_pCudaModule;

CUmodule           cuModNV12toARGB       = 0;
CUfunction         g_kernelNV12toARGB    = 0;
CUfunction         g_kernelPassThru      = 0;

CUcontext          g_oContext = 0;
CUdevice           g_oDevice  = 0;

CUstream           g_ReadbackSID = 0, g_KernelSID = 0;

eColorSpace        g_eColorSpace = ITU601;
float              g_nHue        = 0.0f;

// System Memory surface we want to readback to
BYTE          *g_pFrameYUV[6] = { 0, 0, 0, 0, 0, 0 };
FrameQueue    *g_pFrameQueue   = 0;
VideoSource   *g_pVideoSource  = 0;
VideoParser   *g_pVideoParser  = 0;
VideoDecoder  *g_pVideoDecoder = 0;

ImageGL       *g_pImageGL      = 0; // if we're using OpenGL
CUdeviceptr    g_pInteropFrame[3] = { 0, 0, 0 }; // if we're using CUDA malloc

CUVIDEOFORMAT g_stFormat;

std::string sFileName;

char exec_path[256];

unsigned int g_nWindowWidth  = 0;
unsigned int g_nWindowHeight = 0;

unsigned int g_nClientAreaWidth  = 0;
unsigned int g_nClientAreaHeight = 0;

unsigned int g_nVideoWidth  = 0;
unsigned int g_nVideoHeight = 0;

unsigned int g_FrameCount = 0;
unsigned int g_DecodeFrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;

// Forward declarations
bool initGL(int argc, char **argv, int *pbTCC);
bool initGLTexture(unsigned int nWidth, unsigned int nHeight);
bool loadVideoSource(const char *video_file,
                     unsigned int &width, unsigned int &height,
                     unsigned int &dispWidth, unsigned int &dispHeight);
void initCudaVideo();

void freeCudaResources(bool bDestroyContext);

bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive);
void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
                          CUdeviceptr *ppInteropFrame, size_t pFramePitch,
                          CUmodule cuModNV12toARGB,
                          CUfunction fpCudaKernel, CUstream streamID);
bool drawScene(int field_num);
bool cleanup(bool bDestroyContext);
bool initCudaResources(int argc, char **argv, int *bTCC);

void renderVideoFrame(int bUseInterop);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef bool (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);
PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;

// This allows us to turn off vsync for OpenGL
void setVSync(int interval)
{
    const GLubyte *extensions = glGetString(GL_EXTENSIONS);

    if (strstr((const char *)extensions, "WGL_EXT_swap_control") == 0)
    {
        return;    // Error: WGL_EXT_swap_control extension not supported on your computer.\n");
    }
    else
    {
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");

        if (wglSwapIntervalEXT)
        {
            wglSwapIntervalEXT(interval);
        }
    }
}

#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#else
void setVSync(int interval)
{
}

#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#endif

void printStatistics()
{
    int   hh, mm, ss, msec;

    present_fps = 1.f / (total_time / (g_FrameCount * 1000.f));
    decoded_fps = 1.f / (total_time / (g_DecodeFrameCount * 1000.f));

    msec = ((int)total_time % 1000);
    ss   = (int)(total_time/1000) % 60;
    mm   = (int)(total_time/(1000*60)) % 60;
    hh   = (int)(total_time/(1000*60*60)) % 60;

    printf("\n[%s] statistics\n", sSDKname);
    printf("\t Video Length (hh:mm:ss.msec)   = %02d:%02d:%02d.%03d\n", hh, mm, ss, msec);

    printf("\t Frames Presented (inc repeats) = %d\n", g_FrameCount);
    printf("\t Average Present Rate     (fps) = %4.2f\n", present_fps);

    printf("\t Frames Decoded   (hardware)    = %d\n", g_DecodeFrameCount);
    printf("\t Average Rate of Decoding (fps) = %4.2f\n", decoded_fps);
}

void computeFPS(int bUseInterop)
{
    sdkStopTimer(&frame_timer);

    if (g_bRunning)
    {
        g_fpsCount++;

        if (!(g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty()))
        {
            g_FrameCount++;
        }
    }

    char sFPS[256];
    std::string sDecodeStatus;

    if (g_bDeviceLost)
    {
        sDecodeStatus = "DeviceLost!\0";
        sprintf(sFPS, "%s [%s] - [%s %d]",
                sSDKname, sDecodeStatus.c_str(),
                (g_bIsProgressive ? "Frame" : "Field"),
                g_DecodeFrameCount);

        if (!g_bQAReadback || g_bGLVerify)
        {
            glutSetWindowTitle(sFPS);
        }

        sdkResetTimer(&frame_timer);
        g_fpsCount = 0;
        return;
    }

    if (g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
    {
        sDecodeStatus = "STOP (End of File)\0";

        // we only want to record this once
        if (total_time == 0.0f)
        {
            total_time = sdkGetTimerValue(&global_timer);
        }

        sdkStopTimer(&global_timer);

        if (g_bAutoQuit)
        {
            g_bRunning = false;
            g_bDone    = true;
        }

    }
    else
    {
        if (!g_bRunning)
        {
            sDecodeStatus = "PAUSE\0";
            sprintf(sFPS, "%s [%s] - [%s %d] - Video Display %s / Vsync %s",
                    sAppName, sDecodeStatus.c_str(),
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseDisplay ? "ON" : "OFF",
                    g_bUseVsync   ? "ON" : "OFF");

            if (bUseInterop && (!g_bQAReadback || g_bGLVerify))
            {
                glutSetWindowTitle(sFPS);
            }

        }
        else
        {
            if (g_bFrameStep)
            {
                sDecodeStatus = "STEP\0";
            }
            else
            {
                sDecodeStatus = "PLAY\0";
            }
        }

        if (g_fpsCount == g_fpsLimit)
        {
            float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

            sprintf(sFPS, "[%s] [%s] - [%3.1f fps, %s %d] - Video Display %s / Vsync %s",
                    sAppName, sDecodeStatus.c_str(), ifps,
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseDisplay ? "ON" : "OFF",
                    g_bUseVsync   ? "ON" : "OFF");

            if (bUseInterop)
            {
                glutSetWindowTitle(sFPS);
            }

            printf("[%s] - [%s: %04d, %04.1f fps, time: %04.2f (ms) ]\n",
                   sSDKname, (g_bIsProgressive ? "Frame" : "Field"), g_FrameCount, ifps, 1000.f/ifps);

            sdkResetTimer(&frame_timer);
            g_fpsCount = 0;
        }
    }

    if (g_bDone && g_bAutoQuit && bUseInterop)
    {
        printStatistics();

        cleanup(true);
        exit(EXIT_SUCCESS);
    }

    sdkStartTimer(&frame_timer);
}

bool initCudaResources(int argc, char **argv, int *bTCC)
{
    printf("\n");

    for (int i=0; i < argc; i++)
    {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    printf("\n");

    CUdevice cuda_device;

    // Device is specified at the command line, we need to check if this it TCC or not, and then call the
    // appropriate TCC/WDDM findCudaDevice in order to initialize the CUDA device
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        char name[100];

        cuda_device = getCmdLineArgumentInt(argc, (const char **) argv, "device");
        cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
        checkCudaErrors(cuDeviceGetAttribute(bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, cuda_device));
        checkCudaErrors(cuDeviceGetName(name, 100, cuda_device));
        printf("  -> GPU %d: < %s > driver mode is: %s\n", cuda_device, name, *bTCC ? "TCC" : "WDDM");
        
        // If we detect a TCC device, we always force interop to be off.  Otherwise it is optinal.
        if (*bTCC)
        {
            g_bUseInterop = false;
        }

        if (g_bUseInterop)
        {
            initGL(argc, argv, bTCC);
            cuda_device = findCudaGLDeviceDRV(argc, (const char **)argv);
        }
        else
        {
            cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
        }

        if (cuda_device < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }

        checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
    }
    else
    {
        // If we want to use Graphics Interop, then choose the GPU that is capable
        initGL(argc, argv, bTCC);

        if (*bTCC)
        {
            g_bUseInterop = false;
        }

        if (g_bUseInterop && !(*bTCC))
        {
            cuda_device = findCudaGLDeviceDRV(argc, (const char **)argv);
            checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
        }
        else
        {
            cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
            checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
        }
    }

    // get compute capabilities and the devicename
    int major, minor;
    size_t totalGlobalMem;
    char deviceName[256];
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
    printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
    printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem/(1024*1024));

    // Create CUDA Device w/ GL interop (if WDDM), otherwise CUDA w/o interop (if TCC)
    // (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
    if (g_bUseInterop && !(*bTCC))
    {
        checkCudaErrors(cuGLCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }
    else
    {
        checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }

    // Initialize CUDA related Driver API
    // Determine if we are running on a 32-bit or 64-bit OS and choose the right PTX file
    try
    {
        if (sizeof(void *) == 4)
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi32.ptx", exec_path, 2, 2, 2);
        }
        else
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi64.ptx", exec_path, 2, 2, 2);
        }
    }
    catch (char const *p_file)
    {
        // If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
        printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
        printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
        return false;
    }


    g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi",   &g_kernelNV12toARGB);
    g_pCudaModule->GetCudaFunction("Passthru_drvapi",     &g_kernelPassThru);

    /////////////////Change///////////////////////////
    // Now we create the CUDA resources and the CUDA decoder context
    initCudaVideo();

    if (g_bUseInterop)
    {
        initGLTexture(g_pVideoDecoder->targetWidth(),
                      g_pVideoDecoder->targetHeight());
    }
    else
    {
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[0], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 4));
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[1], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 4));
    }

    CUcontext cuCurrent = NULL;
    CUresult result = cuCtxPopCurrent(&cuCurrent);

    if (result != CUDA_SUCCESS)
    {
        printf("cuCtxPopCurrent: %d\n", result);
        assert(0);
    }

    /////////////////////////////////////////
    return ((g_pCudaModule && g_pVideoDecoder) ? true : false);
}

bool reinitCudaResources()
{
    // Free resources
    cleanup(false);

    // Reinit VideoSource and Frame Queue
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    /////////////////Change///////////////////////////
    initCudaVideo();
    initGLTexture(g_pVideoDecoder->targetWidth(),
                  g_pVideoDecoder->targetHeight());
    /////////////////////////////////////////

    return S_OK;
}

void displayHelp()
{
    printf("\n");
    printf("%s - Help\n\n", sAppName);
    printf("  %s [parameters] [video_file]\n\n", sAppFilename);
    printf("Program parameters:\n");
    printf("\t-i=source.264   - input file for decoding\n");
    printf("\t-o=output.yuv   - specify base Input file for YUV output\n");
    printf("\t-psnr=ref.yuv   - compare PSNR against reference YUV\n");
    printf("\t-decodecuda     - Use CUDA kernels for MPEG-2 (Available with 64+ CUDA cores)\n");
    printf("\t-decodecuvid    - Use NVDEC for MPEG-2, VC-1, H.264, or H.265 decode\n");
    printf("\t-vsync          - Enable vertical sync.\n");
    printf("\t-novsync        - Disable vertical sync.\n");
    printf("\t-repeatframe    - Enable automatic framerate repeating.\n");
    printf("\t-repeatfactor=n - Force repeat every frame n times.\n");
    printf("\t-updateall      - always update CSC matrices.\n");
    printf("\t-displayvideo   - display video frames on the window\n");
    printf("\t-nointerop      - create the CUDA context w/o using graphics interop\n");
    printf("\t-readback       - enable readback of frames to system memory\n");
    printf("\t-device=n       - choose a specific GPU device to decode video with\n");
    printf("\t-nframestart=n  - set the start frame number\n");
    printf("\t-nframeend=n    - set the end frame number\n");
}

void parseCommandLineArguments(int argc, char *argv[])
{
    char video_file[256], yuv_file[256], ref_yuv[256];
    bool bUseDefaultInputFile = true;

    printf("Command Line Arguments:\n");

    for (int n=0; n < argc; n++)
    {
        printf("argv[%d] = %s\n", n, argv[n]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayHelp();
        exit(EXIT_SUCCESS);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "i"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "i", &temp);
        strcpy(video_file, temp);
        bUseDefaultInputFile = false;
    }

    // Search all command file parameters for video files with extensions:
    // mp4, avc, mkv, 264, h264. vc1, wmv, mp2, mpeg2, mpg
    if (checkCmdLineFlag(argc, (const char **)argv, "o"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "o", &temp);
        strcpy(yuv_file, temp);
        g_bReadback = true;
        g_bWriteFile = true;
    }

    // Search all command file parameters for video files with extensions:
    // mp4, avc, mkv, 264, h264. vc1, wmv, mp2, mpeg2, mpg
    if (checkCmdLineFlag(argc, (const char **)argv, "psnr"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "psnr", &temp);
        strcpy(ref_yuv, temp);
        g_bPSNR = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuda"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuvid"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "vsync"))
    {
        g_bUseVsync = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "novsync"))
    {
        g_bUseVsync = false;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "repeatframe"))
    {
        g_bFrameRepeat = true;
        printf("> Framerate Repeating Enabled\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "repeatfactor"))
    {
        g_iRepeatFactor = getCmdLineArgumentInt(argc, (const char **)argv, "repeatfactor");
        printf("g_iRepeatFactor = %d\n", g_iRepeatFactor);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "framestep"))
    {
        g_bFrameStep = true;
        g_bUseDisplay = true;
        g_bUseInterop = true;
        g_fpsLimit = 1;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "updateall"))
    {
        g_bUpdateAll = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "displayvideo"))
    {
        g_bUseDisplay = true;
        g_bUseInterop = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "nointerop"))
    {
        g_bUseInterop = false;
        printf("NVDECODE/OpenGL graphics interop disabled\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "readback"))
    {
        g_bReadback = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        g_DeviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "nframestart"))
    {
        g_nFrameStart = getCmdLineArgumentInt(argc, (const char **)argv, "nframestart");
        printf("YUV output @ nStartFrame = %d\n", g_nFrameStart);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "nframeend"))
    {
        g_nFrameEnd = getCmdLineArgumentInt(argc, (const char **)argv, "nframeend");
        printf("YUV output @ nStartEnd = %d\n", g_nFrameEnd);
    }

    if (g_bUseDisplay == false)
    {
        g_bQAReadback = true;
        g_bUseInterop = false;
    }

    if (g_bLoop == false)
    {
        g_bAutoQuit = true;
    }
    if (bUseDefaultInputFile)
    {
        strcpy(video_file, sdkFindFilePath(VIDEO_SOURCE_FILE, argv[0]));
    }


    // We load the default video file for the SDK sample
    if (bUseDefaultInputFile)
    {
        strcpy(video_file, sdkFindFilePath(VIDEO_SOURCE_FILE, argv[0]));
    }

    // Now verify the video file is legit
    FILE *fp = NULL;
    FOPEN(fp, video_file, "r");
    if (fp == NULL)
    {
        printf("[%s]: unable to find file: [%s]\nExiting...\n", sAppFilename, VIDEO_SOURCE_FILE);
        exit(EXIT_FAILURE);
    }

    if (fp)
    {
        fclose(fp);
    }

    // Now verify the input reference YUV file is legit
    FOPEN(fpRefYUV, ref_yuv, "r");
    if (ref_yuv == NULL && fpRefYUV == NULL)
    {
        printf("[%s]: unable to find file: [%s]\nExiting...\n", sAppFilename, ref_yuv);
        exit(EXIT_FAILURE);
    }

    // default video file loaded by this sample
    sFileName = video_file;

    if (g_bWriteFile && strlen(yuv_file) > 0)
    {
        printf("[%s]: output file: [%s]\n", sAppFilename, yuv_file);

        FOPEN(fpWriteYUV, yuv_file, "wb");
        if (fpWriteYUV == NULL)
        {
            printf("Error opening file [%s]\n", yuv_file);
        }
    }

    // store the current path so we can reinit the CUDA context
    strcpy(exec_path, argv[0]);

    printf("[%s]: input file:  [%s]\n", sAppFilename, video_file);
}

void SaveFrameAsYUV(unsigned char *pdst,
    const unsigned char *psrc,
    int width, int height, int pitch)
{
    int x, y, width_2, height_2;
    int xy_offset = width*height;
    int uvoffs = (width / 2)*(height / 2);
    const unsigned char *py = psrc;
    const unsigned char *puv = psrc + height*pitch;

    if (((long)g_DecodeFrameCount >= g_nFrameStart) &&
        ((long)g_DecodeFrameCount <= g_nFrameEnd)
        )
    {
        //      printf(" Saving YUV Frame %d (start,end)=(%d,%d)\n", g_DecodeFrameCount, g_nFrameStart, g_nFrameEnd);
        printf("%d+", g_DecodeFrameCount);
    }
    else if ((g_nFrameStart == -1) && (g_nFrameEnd == -1))
    {
        printf("+");
    }
    else // we do nothing and exit
    {
        return;
    }

    // luma
    for (y = 0; y<height; y++)
    {
        memcpy(&pdst[y*width], py, width);
        py += pitch;
    }

    // De-interleave chroma
    width_2 = width >> 1;
    height_2 = height >> 1;
    for (y = 0; y<height_2; y++)
    {
        for (x = 0; x<width_2; x++)
        {
            pdst[xy_offset + y*(width_2)+x] = puv[x * 2];
            pdst[xy_offset + uvoffs + y*(width_2)+x] = puv[x * 2 + 1];
        }
        puv += pitch;
    }

    fwrite(pdst, 1, width*height + (width*height) / 2, fpWriteYUV);
}

int main(int argc, char *argv[])
{
    printf("[%s]\n", sAppName);

    sdkCreateTimer(&frame_timer);
    sdkResetTimer(&frame_timer);

    sdkCreateTimer(&global_timer);
    sdkResetTimer(&global_timer);

    // parse the command line arguments
    parseCommandLineArguments(argc, argv);

    // Initialize the CUDA and NVDECODE
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    typedef HMODULE CUDADRIVER;
#else
    typedef void *CUDADRIVER;
#endif
    CUDADRIVER hHandleDriver = 0;
    cuInit   (0, __CUDA_API_VERSION, hHandleDriver);
    cuvidInit(0);

    // Find out the video size (uses NVDECODE calls)
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    // Determine the proper window size needed to create the correct *client* area
    // that is of the size requested by m_dimensions.
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    RECT adjustedWindowSize;
    DWORD dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
    SetRect(&adjustedWindowSize, 0, 0, g_nVideoWidth  , g_nVideoHeight);
    AdjustWindowRect(&adjustedWindowSize, dwWindowStyle, false);
#endif

    g_nVideoWidth   = PAD_ALIGN(g_nVideoWidth   , 0x3F);
    g_nVideoHeight  = PAD_ALIGN(g_nVideoHeight  , 0x0F);

    // Initialize CUDA and try to connect with an OpenGL context
    // Other video memory resources will be available
    int bTCC = 0;

    if (initCudaResources(argc, argv, &bTCC) == false)
    {
        g_bAutoQuit  = true;
        g_bException = true;
        g_bWaived    = true;
        goto ExitApp;
    }

    g_pVideoSource->start();
    g_bRunning = true;

    sdkStartTimer(&global_timer);
    sdkResetTimer(&global_timer);

    if (!g_bUseInterop)
    {
        // On this case we drive the display with a while loop (no openGL calls)
        while (!g_bDone)
        {
            renderVideoFrame(g_bUseInterop);
        }
    }
    else
    {
        glutMainLoop();
    }

    // we only want to record this once
    if (total_time == 0.0f)
    {
        total_time = sdkGetTimerValue(&global_timer);
    }
    sdkStopTimer(&global_timer);

    g_pFrameQueue->endDecode();
    g_pVideoSource->stop();

    if (fpWriteYUV != NULL)
    {
        fflush(fpWriteYUV);
        fclose(fpWriteYUV);
        fpWriteYUV = NULL;
    }

    printStatistics();

ExitApp:
    // clean up CUDA and OpenGL resources
    cleanup(g_bWaived ? false : true);

    if (g_bWaived)
    {
        exit(EXIT_WAIVED);
    }
    else
    {
        exit(g_bException ? EXIT_FAILURE : EXIT_SUCCESS);
    }

    return 0;
}


// display results using OpenGL
void display()
{
    renderVideoFrame(true);
}


void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            if (g_pFrameQueue)
            {
                g_pFrameQueue->endDecode();
            }

            if (g_pVideoSource)
            {
                g_pVideoSource->stop();
            }

            printStatistics();
            cleanup(true);
            exit(EXIT_FAILURE);
            break;

        case 'F':
        case 'f':
            g_bLinearFiltering = !g_bLinearFiltering;

            if (g_pImageGL)
                g_pImageGL->setTextureFilterMode(g_bLinearFiltering ? GL_LINEAR : GL_NEAREST,
                                                 g_bLinearFiltering ? GL_LINEAR : GL_NEAREST);

            break;

        case ' ':
            g_bRunning = !g_bRunning;
            break;

        default:
            break;
    }

    glutPostRedisplay();
}

void idle()
{
    glutPostRedisplay();
}

void reshape(int window_x, int window_y)
{
    printf("reshape() glViewport(%d, %d, %d, %d)\n", 0, 0, window_x, window_y);

    glViewport(0, 0, window_x, window_y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

// Initialize OpenGL Resources
bool initGL(int argc, char **argv, int *pbTCC)
{
    int dev, device_count = 0;
    bool bSpecifyDevice=false;
    char device_name[256];

    // Check for a min spec of Compute 1.1 capability before running
    checkCudaErrors(cuDeviceGetCount(&device_count));

    for (int i=0; i < argc; i++)
    {
        int string_start = 0;

        while (argv[i][string_start++] != '-');

        const char *string_argv = &argv[i][string_start];

        if (!STRNCASECMP(string_argv, "device=", 7))
        {
            bSpecifyDevice = true;
        }
    }

    // If deviceID == 0, and there is more than 1 device, let's find the first available graphics GPU
    if (!bSpecifyDevice && device_count > 0)
    {
        for (int i=0; i < device_count; i++)
        {
            checkCudaErrors(cuDeviceGet(&dev, i));
            checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

            int bSupported = checkCudaCapabilitiesDRV(1, 1, i);

            if (!bSupported)
            {
                printf("  -> GPU: \"%s\" does not meet the minimum spec of SM 1.1\n", device_name);
                printf("  -> A GPU with a minimum compute capability of SM 1.1 or higher is required.\n");
                return false;
            }

#if CUDA_VERSION >= 3020
            checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
            printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

            if (*pbTCC)
            {
                continue;
            }
            else
            {
                g_DeviceID = i; // we choose an available WDDM display device
            }

#else

            // We don't know for sure if this is a TCC device or not, if it is Tesla we will not run
            if (!STRNCASECMP(device_name, "Tesla", 5))
            {
                printf("  \"%s\" does not support %s\n", device_name, sSDKname);
                *pbTCC = 1;
                return false;
            }
            else
            {
                *pbTCC = 0;
            }

#endif
            printf("\n");
        }
    }
    else
    {
        if ((g_DeviceID > (device_count-1)) || (g_DeviceID < 0))
        {
            printf(" >>> Invalid GPU Device ID=%d specified, only %d GPU device(s) are available.<<<\n", g_DeviceID, device_count);
            printf(" >>> Valid GPU ID (n) range is between [%d,%d]...  Exiting... <<<\n", 0, device_count-1);
            return false;
        }

        // We are specifying a GPU device, check to see if it is TCC or not
        checkCudaErrors(cuDeviceGet(&dev, g_DeviceID));
        checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

#if CUDA_VERSION >= 3020
        checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
        printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");
#else

        // We don't know for sure if this is a TCC device or not, if it is Tesla we will not run
        if (!STRNCASECMP(device_name, "Tesla", 5))
        {
            printf("  \"%s\" does not support %s\n", device_name, sSDKname);
            *pbTCC = 1;
            return false;
        }
        else
        {
            *pbTCC = 0;
        }

#endif
    }

    if (!(*pbTCC))
    {
        // initialize GLUT
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(g_nWindowWidth, g_nWindowHeight);
        glutCreateWindow(sAppName);
        reshape(g_nWindowWidth, g_nWindowHeight);

        printf(">> initGL() creating window [%d x %d]\n", g_nWindowWidth, g_nWindowHeight);

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutKeyboardFunc(keyboard);
        glutIdleFunc(idle);

        glewInit();

        if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
        {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 1.5\n");
            fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
            fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            return true;
        }

        setVSync(g_bUseVsync ? 1 : 0);
    }
    else
    {
        fprintf(stderr, "> %s is decoding w/o visualization\n", sSDKname);
    }

    return true;
}


// Initializes OpenGL Textures (allocation and initialization)
bool
initGLTexture(unsigned int nWidth, unsigned int nHeight)
{
    g_pImageGL = new ImageGL(nWidth, nHeight,
                             nWidth, nHeight,
                             g_bUseVsync,
                             ImageGL::BGRA_PIXEL_FORMAT);
    g_pImageGL->clear(0x80);

    g_pImageGL->setCUDAcontext(g_oContext);
    g_pImageGL->setCUDAdevice(g_oDevice);
    return true;
}


bool
loadVideoSource(const char *video_file,
                unsigned int &width    , unsigned int &height,
                unsigned int &dispWidth, unsigned int &dispHeight)
{
    std::auto_ptr<FrameQueue> apFrameQueue(new FrameQueue);
    std::auto_ptr<VideoSource> apVideoSource(new VideoSource(video_file, apFrameQueue.get()));

    // retrieve the video source (width,height)
    apVideoSource->getSourceDimensions(width, height);
    apVideoSource->getSourceDimensions(dispWidth, dispHeight);

    memset(&g_stFormat, 0, sizeof(CUVIDEOFORMAT));
    std::cout << (g_stFormat = apVideoSource->format()) << std::endl;

    if (g_bFrameRepeat)
    {
        if (apVideoSource->format().frame_rate.denominator > 0)
        {
            g_iRepeatFactor = (int)(60.0f / ceil((float)apVideoSource->format().frame_rate.numerator / (float)apVideoSource->format().frame_rate.denominator));
        }
    }

    printf("Frame Rate Playback Speed = %d fps\n", 60 / g_iRepeatFactor);

    g_pFrameQueue  = apFrameQueue.release();
    g_pVideoSource = apVideoSource.release();

    if (g_pVideoSource->format().codec == cudaVideoCodec_JPEG)
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    bool IsProgressive = 0;
    g_pVideoSource->getProgressive(IsProgressive);
    return IsProgressive;
}

void
initCudaVideo()
{
    // bind the context lock to the CUDA context
    CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);
    CUVIDEOFORMATEX oFormatEx;
    memset(&oFormatEx, 0, sizeof(CUVIDEOFORMATEX));
    oFormatEx.format = g_stFormat;

    if (result != CUDA_SUCCESS)
    {
        printf("cuvidCtxLockCreate failed: %d\n", result);
        assert(0);
    }

    std::auto_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(g_pVideoSource->format(), g_oContext, g_eVideoCreateFlags, g_CtxLock));
    std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue, &oFormatEx, &g_oContext));
    g_pVideoSource->setParser(*apVideoParser.get());

    g_pVideoParser  = apVideoParser.release();
    g_pVideoDecoder = apVideoDecoder.release();

    // Create a Stream ID for handling Readback
    if (g_bReadback)
    {
        checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
        checkCudaErrors(cuStreamCreate(&g_KernelSID,   0));
        printf(">> initCudaVideo()\n");
        printf("   CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
        printf("   CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID   == 0) ? "Disabled" : "Enabled"), g_KernelSID);
    }
}


void
freeCudaResources(bool bDestroyContext)
{
    if (g_pVideoParser)
    {
        delete g_pVideoParser;
    }

    if (g_pVideoDecoder)
    {
        delete g_pVideoDecoder;
    }

    if (g_pVideoSource)
    {
        delete g_pVideoSource;
    }

    if (g_pFrameQueue)
    {
        delete g_pFrameQueue;
    }

    if (g_ReadbackSID)
    {
        checkCudaErrors(cuStreamDestroy(g_ReadbackSID));
    }

    if (g_KernelSID)
    {
        checkCudaErrors(cuStreamDestroy(g_KernelSID));
    }

    if (g_CtxLock)
    {
        checkCudaErrors(cuvidCtxLockDestroy(g_CtxLock));
    }

    if (g_oContext && bDestroyContext)
    {
        checkCudaErrors(cuCtxDestroy(g_oContext));
        g_oContext = NULL;
    }
}

// Run the Cuda part of the computation (if g_pFrameQueue is empty, then return false)
bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive)
{
    CUVIDPARSERDISPINFO oDisplayInfo;

    if (g_pFrameQueue->dequeue(&oDisplayInfo))
    {
        CCtxAutoLock lck(g_CtxLock);
        // Push the current CUDA context (only if we are using CUDA decoding path)
        cuCtxPushCurrent(g_oContext);

        CUdeviceptr  pDecodedFrame[3] = { 0, 0, 0 };
        CUdeviceptr  pInteropFrame[3] = { 0, 0, 0 };

        *pbIsProgressive = oDisplayInfo.progressive_frame;
        g_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;

        int num_fields = 1;
        if (g_bUseVsync) {
            num_fields = std::min(2 + oDisplayInfo.repeat_first_field, 3);            
        }
        nRepeats = num_fields;

        CUVIDPROCPARAMS oVideoProcessingParameters;
        memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

        oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;        
        oVideoProcessingParameters.top_field_first = oDisplayInfo.top_field_first;
        oVideoProcessingParameters.unpaired_field = (oDisplayInfo.repeat_first_field < 0);

        for (int active_field = 0; active_field < num_fields; active_field++) {
            unsigned int nDecodedPitch = 0;
            unsigned int nWidth = 0;
            unsigned int nHeight = 0;

            oVideoProcessingParameters.second_field = active_field;

            // map decoded video frame to CUDA surfae
            if (g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters) != CUDA_SUCCESS)
            {
                // release the frame, so it can be re-used in decoder
                g_pFrameQueue->releaseFrame(&oDisplayInfo);

                // Detach from the Current thread
                checkCudaErrors(cuCtxPopCurrent(NULL));

                return false;
            }
            nWidth = g_pVideoDecoder->targetWidth(); // PAD_ALIGN(g_pVideoDecoder->targetWidth(), 0x3F);
            nHeight = g_pVideoDecoder->targetHeight(); // PAD_ALIGN(g_pVideoDecoder->targetHeight(), 0x0F);
            // map OpenGL PBO or CUDA memory
            size_t nTexturePitch = 0;

            // If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
            if (g_bReadback && g_bFirstFrame && g_ReadbackSID)
            {
                CUresult result;
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[0], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[1], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[2], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[3], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[4], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[5], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));

                g_bFirstFrame = false;

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

            // If streams are enabled, we can perform the readback to the host while the kernel is executing
            if (g_bReadback && g_ReadbackSID)
            {
                CUresult result = cuMemcpyDtoHAsync(g_pFrameYUV[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), g_ReadbackSID);

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

#if ENABLE_DEBUG_OUT
            printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
                   (oDisplayInfo.progressive_frame ? "Frame" : "Field"),
                   g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

            if (g_pImageGL)
            {
                // map the texture surface
                g_pImageGL->map(&pInteropFrame[active_field], &nTexturePitch, active_field);
                nTexturePitch /= g_pVideoDecoder->targetHeight();
            }
            else
            {
                pInteropFrame[active_field] = g_pInteropFrame[active_field];
                nTexturePitch = g_pVideoDecoder->targetWidth() * 2;
            }

            // perform post processing on the CUDA surface (performs colors space conversion and post processing)
            // comment this out if we inclue the line of code seen above

            cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field],
                                 nTexturePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);

            if (g_pImageGL)
            {
                // unmap the texture surface
                g_pImageGL->unmap(active_field);
            }

            // unmap video frame
            // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
            g_pVideoDecoder->unmapFrame(pDecodedFrame[active_field]);
            g_DecodeFrameCount++;

            if (g_bWriteFile)
            {
                checkCudaErrors(cuStreamSynchronize(g_ReadbackSID));
                SaveFrameAsYUV(g_pFrameYUV[active_field + 3],
                    g_pFrameYUV[active_field],
                    nWidth, nHeight, nDecodedPitch);
            }
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
        // release the frame, so it can be re-used in decoder
        g_pFrameQueue->releaseFrame(&oDisplayInfo);         
    }
    else
    {
        // Frame Queue has no frames, we don't compute FPS until we start
        return false;
    }

    // check if decoding has come to an end.
    // if yes, signal the app to shut down.
    if (!g_pVideoSource->isStarted() && g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
    {
        // Let's free the Frame Data
        if (g_ReadbackSID)
        {
            cuMemFreeHost((void *)g_pFrameYUV[0]);
            cuMemFreeHost((void *)g_pFrameYUV[1]);
            cuMemFreeHost((void *)g_pFrameYUV[2]);
            cuMemFreeHost((void *)g_pFrameYUV[3]);
            cuMemFreeHost((void *)g_pFrameYUV[4]);
            cuMemFreeHost((void *)g_pFrameYUV[5]);

            g_pFrameYUV[0] = NULL;
            g_pFrameYUV[1] = NULL;
            g_pFrameYUV[2] = NULL;
            g_pFrameYUV[3] = NULL;
            g_pFrameYUV[4] = NULL;
            g_pFrameYUV[5] = NULL;
        }

        // Let's just stop, and allow the user to quit, so they can at least see the results
        g_pVideoSource->stop();

        // If we want to loop reload the video file and restart
        if (g_bLoop && !g_bAutoQuit)
        {
            reinitCudaResources();
            g_FrameCount = 0;
            g_DecodeFrameCount = 0;
            g_pVideoSource->start();
        }

        if (g_bAutoQuit)
        {
            g_bDone = true;
        }
    }

    return true;
}

// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
                     CUdeviceptr *ppTextureData,  size_t nTexturePitch,
                     CUmodule cuModNV12toARGB,
                     CUfunction fpCudaKernel, CUstream streamID)
{
    uint32 nWidth  = g_pVideoDecoder->targetWidth();
    uint32 nHeight = g_pVideoDecoder->targetHeight();

    // Upload the Color Space Conversion Matrices
    if (g_bUpdateCSC)
    {
        // CCIR 601/709
        float hueColorSpaceMat[9];
        setColorSpaceMatrix(g_eColorSpace,    hueColorSpaceMat, g_nHue);
        updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);

        if (!g_bUpdateAll)
        {
            g_bUpdateCSC = false;
        }
    }

    // TODO: Stage for handling video post processing

    // Final Stage: NV12toARGB color space conversion
    cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch,
                                      *ppTextureData, nTexturePitch,
                                      nWidth, nHeight, fpCudaKernel, streamID);
}

// Draw the final result on the screen
bool drawScene(int field_num)
{
    bool hr = true;

    // Normal OpenGL rendering code
    // render image
    if (g_pImageGL)
    {
        g_pImageGL->render(field_num);
    }

    return hr;
}

// Release all previously initd objects
bool cleanup(bool bDestroyContext)
{
    if (fpWriteYUV != NULL)
    {
        fflush(fpWriteYUV);
        fclose(fpWriteYUV);
        fpWriteYUV = NULL;
    }

    if (bDestroyContext)
    {
        // Attach the CUDA Context (so we may properly free memroy)
        checkCudaErrors(cuCtxPushCurrent(g_oContext));

        if (g_pInteropFrame[0])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[0]));
        }

        if (g_pInteropFrame[1])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[1]));
        }

        if (g_pInteropFrame[2])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[2]));
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
    }

    if (g_pImageGL)
    {
        delete g_pImageGL;
        g_pImageGL = NULL;
    }

    freeCudaResources(bDestroyContext);

    return true;
}

// Launches the CUDA kernels to fill in the texture data
void renderVideoFrame(int bUseInterop)
{
    static unsigned int nRepeatFrame = 0;
    int repeatFactor = g_iRepeatFactor;
    int bIsProgressive = 1, bFPSComputed = 0;
    bool bFramesDecoded = false;

    if (0 != g_pFrameQueue)
    {
        // if not running, we simply don't copy new frames from the decoder
        if (!g_bDeviceLost && g_bRunning)
        {
            bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, bUseInterop, &bIsProgressive);
        }
    }
    else
    {
        return;
    }

    if (bFramesDecoded)
    {
        while (repeatFactor-- > 0)
        {
            if (g_bUseDisplay && bUseInterop)
            {
                for (int i = 0; i < nRepeatFrame; i++) {
                    drawScene(i);
                    glutSwapBuffers();

                    if (!repeatFactor)
                    {
                        computeFPS(bUseInterop);
                    }
                }

                bFPSComputed = 1;
            }

            // Pass the Windows handle to show Frame Rate on the window title
            if (!bFPSComputed && !repeatFactor)
            {
                computeFPS(bUseInterop);
            }

            if (g_bUseDisplay && bUseInterop)
            {
                glutReportErrors();
            }

        }
    }

    if (bFramesDecoded && g_bFrameStep)
    {
        if (g_bRunning)
        {
            g_bRunning = false;
        }
    }
}

