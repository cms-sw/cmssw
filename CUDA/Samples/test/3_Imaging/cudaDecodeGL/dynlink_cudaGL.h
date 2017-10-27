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
 
#ifndef CUDAGL_H
#define CUDAGL_H

#if INIT_CUDA_GL

#ifndef __CUDA_API_VERSION
#define __CUDA_API_VERSION 4000
#endif

#include <GL/glew.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file dynlink_cudaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 */

/**
 * \defgroup CUDA_GL OpenGL Interoperability
 * \ingroup CUDA_DRIVER
 *
 * ___MANBRIEF___ OpenGL interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface. Note that mapping 
 * of OpenGL resources is performed with the graphics API agnostic, resource 
 * mapping interface described in \ref CUDA_GRAPHICS "Graphics Interoperability".
 *
 * @{
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#if !defined(WGL_NV_gpu_affinity)
typedef void* HGPUNV;
#endif
#endif /* _WIN32 */

typedef CUresult CUDAAPI tcuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
typedef CUresult CUDAAPI tcuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef CUresult CUDAAPI tcuWGLGetDevice(CUdevice *pDevice, HGPUNV hGpu);
#endif /* _WIN32 */

/**
 * CUDA devices corresponding to an OpenGL device
 */
typedef enum CUGLDeviceList_enum {
    CU_GL_DEVICE_LIST_ALL            = 0x01, /**< The CUDA devices for all GPUs used by the current OpenGL context */
    CU_GL_DEVICE_LIST_CURRENT_FRAME  = 0x02, /**< The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame */
    CU_GL_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame */
} CUGLDeviceList;

#if __CUDA_API_VERSION >= 6050
typedef CUresult CUDAAPI tcuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
#endif /* __CUDA_API_VERSION >= 6050 */

/**
 * \defgroup CUDA_GL_DEPRECATED OpenGL Interoperability [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated OpenGL interoperability functions of the low-level
 * CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated OpenGL interoperability functionality.
 *
 * @{
 */

/** Flags to map or unmap a resource */
typedef enum CUGLmap_flags_enum {
    CU_GL_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,    
} CUGLmap_flags;

//#if __CUDA_API_VERSION >= 3020
typedef CUresult CUDAAPI tcuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, CUdevice device);
//#endif /* __CUDA_API_VERSION >= 3020 */

typedef CUresult CUDAAPI tcuGLInit(void);
typedef CUresult CUDAAPI tcuGLRegisterBufferObject(GLuint buffer);

#if __CUDA_API_VERSION >= 3020
typedef CUresult CUDAAPI tcuGLMapBufferObject(CUdeviceptr *dptr, size_t *size, GLuint buffer);
#endif /* __CUDA_API_VERSION >= 3020 */

typedef CUresult CUDAAPI tcuGLUnmapBufferObject(GLuint buffer);
typedef CUresult CUDAAPI tcuGLUnregisterBufferObject(GLuint buffer);
typedef CUresult CUDAAPI tcuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags);

#if __CUDA_API_VERSION >= 3020
typedef CUresult CUDAAPI tcuGLMapBufferObjectAsync(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream);
#endif /* __CUDA_API_VERSION >= 3020 */

typedef CUresult CUDAAPI tcuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream);
typedef CUresult CUDAAPI tcuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
extern tcuWGLGetDevice                 *cuWGLGetDevice;
#endif

extern tcuGLCtxCreate                  *cuGLCtxCreate;
extern tcuGLCtxCreate                  *cuGLCtxCreate_v2;
extern tcuGLMapBufferObject            *cuGLMapBufferObject;
extern tcuGLMapBufferObject            *cuGLMapBufferObject_v2;
extern tcuGLMapBufferObjectAsync       *cuGLMapBufferObjectAsync;

#if __CUDA_API_VERSION >= 6050
extern tcuGLGetDevices                 *cuGLGetDevices;
#endif

extern tcuGraphicsGLRegisterBuffer     *cuGraphicsGLRegisterBuffer;
extern tcuGraphicsGLRegisterImage      *cuGraphicsGLRegisterImage;
extern tcuGLSetBufferObjectMapFlags    *cuGLSetBufferObjectMapFlags;
extern tcuGLRegisterBufferObject       *cuGLRegisterBufferObject;

extern tcuGLUnmapBufferObject          *cuGLUnmapBufferObject;
extern tcuGLUnmapBufferObjectAsync     *cuGLUnmapBufferObjectAsync;

extern tcuGLUnregisterBufferObject     *cuGLUnregisterBufferObject;
extern tcuGLGetDevices                 *cuGLGetDevices; // CUDA 6.5 only


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <Windows.h>
typedef HMODULE CUDADRIVER;
#else
typedef void *CUDADRIVER;
#endif



/************************************/
CUresult CUDAAPI cuInitGL(unsigned int, int cudaVersion, CUDADRIVER &CudaDrvLib);
/************************************/

#ifdef __cplusplus
};
#endif

#endif

#endif
