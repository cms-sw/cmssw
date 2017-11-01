/*
 * eglstrm_common.h
 *
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//
// DESCRIPTION:   Common EGL stream functions header file
//

#ifndef _EGLSTRM_COMMON_H_
#define _EGLSTRM_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "cuda.h"
#include "cudaEGL.h"

#include <time.h>
#define TIME_DIFF(end, start) (getMicrosecond(end) - getMicrosecond(start))

extern EGLStreamKHR g_producerEglStream;
extern EGLStreamKHR g_consumerEglStream;
extern EGLDisplay g_producerEglDisplay;
extern EGLDisplay g_consumerEglDisplay;
extern bool verbose;

#define EXTENSION_LIST(T) \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR ) \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR ) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR ) \
    T( PFNEGLQUERYSTREAMU64KHRPROC,        eglQueryStreamu64KHR ) \
    T( PFNEGLQUERYSTREAMTIMEKHRPROC,       eglQueryStreamTimeKHR ) \
    T( PFNEGLSTREAMATTRIBKHRPROC,          eglStreamAttribKHR ) \
    T( PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR ) \
    T( PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR ) \
    T( PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, \
                                    eglStreamConsumerGLTextureExternalKHR ) \
    T( PFNEGLQUERYDEVICESEXTPROC, eglQueryDevicesEXT ) \
    T( PFNEGLGETPLATFORMDISPLAYEXTPROC, eglGetPlatformDisplayEXT ) \
    T( PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC, eglGetStreamFileDescriptorKHR) \
    T( PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC, eglCreateStreamFromFileDescriptorKHR)


#define EXTLST_DECL(tx, x)  tx x = NULL;
#define EXTLST_EXTERN(tx, x) extern tx x;
#define EXTLST_ENTRY(tx, x) { (extlst_fnptr_t *)&x, #x },


#define MAX_STRING_SIZE     256
#define INIT_DATA           0x01
#define PROD_DATA           0x07
#define CONS_DATA           0x04

#define SOCK_PATH       "/tmp/tegra_sw_egl_socket"

typedef struct _TestArgs {
    unsigned int charCnt;
    bool isProducer;
} TestArgs;


extern int WIDTH, HEIGHT;

int eglSetupExtensions(bool is_dgpu);
void PrintEGLStreamState(EGLint streamState);
int EGLStreamInit(bool isCrossDevice, int isConsumer, EGLNativeFileDescriptorKHR fileDesc);
void EGLStreamFini(void);

int EGLStreamSetAttr(EGLDisplay display, EGLStreamKHR eglStream);
int UnixSocketConnect(const char *socket_name);
int EGLStreamSendfd(int send_fd, int fd_to_send);
int UnixSocketCreate(const char *socket_name);
int EGLStreamReceivefd(int connect_fd);

static clockid_t clock_id =   CLOCK_MONOTONIC; //CLOCK_PROCESS_CPUTIME_ID;
static double getMicrosecond(struct timespec t)
{
    return ((t.tv_sec) * 1000000.0 + (t.tv_nsec) / 1.0e3 );
}

static inline void getTime(struct timespec *t)
{
    clock_gettime(clock_id, t);
}
#endif
