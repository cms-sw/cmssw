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

#include "ImageGL.h"

#include "dynlink_cuda.h"   // <cuda.h>
#include "dynlink_cudaGL.h" // <cudaGL.h>

#include <cassert>

#include "helper_cuda_drvapi.h"

GLuint compile_glsl_shader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

ImageGL::ImageGL(unsigned int nDispWidth,
                 unsigned int nDispHeight,
                 unsigned int nTexWidth,
                 unsigned int nTexHeight,
                 bool bVsync, 
                 PixelFormat ePixelFormat)
    : nWidth_(nDispWidth)
    , nHeight_(nDispHeight)
    , nTexWidth_(nTexWidth)
    , nTexHeight_(nTexHeight)
    , e_PixFmt_(ePixelFormat)
    , bVsync_(bVsync) 
    , bIsCudaResource_(false)
{
    int nFrames = bVsync_ ? 3 : 1;

    glGenBuffers(nFrames, gl_pbo_);

    for (int n=0; n < nFrames; n++)
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo_[n]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, nTexWidth*nTexHeight*4, NULL, GL_STREAM_DRAW_ARB);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        registerAsCudaResource(n);
    }

    // create texture for display
    glGenTextures(nFrames, gl_texid_);

    // setup the Texture filtering mode
    setTextureFilterMode(GL_NEAREST, GL_NEAREST);

    // load gl_shader_ program
    gl_shader_ = compile_glsl_shader(GL_FRAGMENT_PROGRAM_ARB, gl_shader_code);
}

ImageGL::~ImageGL()
{
    int nFrames = bVsync_ ? 3 : 1;

    for (int n=0; n < nFrames; n++)
    {
        unregisterAsCudaResource(n);
    }

    glDeleteBuffers(nFrames, gl_pbo_);
    glDeleteTextures(nFrames, gl_texid_);
    glDeleteProgramsARB(1, &gl_shader_);
}

void
ImageGL::registerAsCudaResource(int field_num)
{
    // register the OpenGL resources that we'll use within CD
    checkCudaErrors(cuGLRegisterBufferObject(gl_pbo_[field_num]));
    getLastCudaDrvErrorMsg("cuGLRegisterBufferObject (gl_pbo_) failed");
    bIsCudaResource_ = true;
}

void
ImageGL::unregisterAsCudaResource(int field_num)
{
    cuCtxPushCurrent(oContext_);
    checkCudaErrors(cuGLUnregisterBufferObject(gl_pbo_[field_num]));
    bIsCudaResource_ = false;
    cuCtxPopCurrent(NULL);
}

void
ImageGL::setTextureFilterMode(GLuint nMINfilter, GLuint nMAGfilter)
{
    int nFrames = bVsync_ ? 3 : 1;

    printf("setTextureFilterMode(%s,%s)\n",
           (nMINfilter == GL_NEAREST) ? "GL_NEAREST" : "GL_LINEAR",
           (nMAGfilter == GL_NEAREST) ? "GL_NEAREST" : "GL_LINEAR");

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        glBindTexture(GL_TEXTURE_TYPE, gl_texid_[field_num]);
        glTexImage2D(GL_TEXTURE_TYPE, 0, GL_RGBA8, nTexWidth_, nTexHeight_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, nMINfilter);
        glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, nMAGfilter);
        glBindTexture(GL_TEXTURE_TYPE, 0);
    }
}

void
ImageGL::setCUDAcontext(CUcontext oContext)
{
    oContext_ = oContext;
    printf("ImageGL::CUcontext = %08lx\n", (unsigned long)oContext);
}

void
ImageGL::setCUDAdevice(CUdevice oDevice)
{
    oDevice_ = oDevice;
    printf("ImageGL::CUdevice  = %08lx\n", (unsigned long)oDevice);
}

bool
ImageGL::isCudaResource()
const
{
    return bIsCudaResource_;
}

void
ImageGL::map(CUdeviceptr *pImageData, size_t *pImagePitch, int field_num)
{
    checkCudaErrors(cuGLMapBufferObject(pImageData, pImagePitch, gl_pbo_[field_num]));
    assert(0 != *pImagePitch);
}

void
ImageGL::unmap(int field_num)
{
    checkCudaErrors(cuGLUnmapBufferObject(gl_pbo_[field_num]));
}

void
ImageGL::clear(unsigned char nClearColor)
{
    // Can only be cleared if surface is a CUDA resource
    assert(bIsCudaResource_);

    int nFrames = bVsync_ ? 3 : 1;
    size_t       imagePitch;
    CUdeviceptr  pImageData;

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        map(&pImageData, &imagePitch, field_num);
        // clear the surface to solid white
        checkCudaErrors(cuMemsetD8(pImageData, nClearColor, nTexWidth_*nTexHeight_* Bpp()));
        unmap(field_num);
    }
}

void
ImageGL::render(int field_num)
const
{
    // Common display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // load texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo_[field_num]);
        glBindTexture(GL_TEXTURE_TYPE, gl_texid_[field_num]);
        glTexSubImage2D(GL_TEXTURE_TYPE, 0, 0, 0, nTexWidth_, nTexHeight_, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_shader_);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        float fTexWidth = (float)nWidth_ / (float)nTexWidth_;
        float fTexHeight= (float)nHeight_/ (float)nTexHeight_;

        glBegin(GL_QUADS);

        if (GL_TEXTURE_TYPE == GL_TEXTURE_2D)
        {
            glTexCoord2f(0        , fTexHeight);
            glVertex2f(0, 0);
            glTexCoord2f(fTexWidth, fTexHeight);
            glVertex2f(1, 0);
            glTexCoord2f(fTexWidth, 0);
            glVertex2f(1, 1);
            glTexCoord2f(0        , 0);
            glVertex2f(0, 1);
        }
        else
        {
            glTexCoord2f(0         , (GLfloat) nTexHeight_);
            glVertex2f(0, 0);
            glTexCoord2f((GLfloat) nTexWidth_, (GLfloat) nTexHeight_);
            glVertex2f(1, 0);
            glTexCoord2f((GLfloat) nTexWidth_, 0);
            glVertex2f(1, 1);
            glTexCoord2f(0         , 0);
            glVertex2f(0, 1);
        }

        glEnd();
        glBindTexture(GL_TEXTURE_TYPE, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }
}
