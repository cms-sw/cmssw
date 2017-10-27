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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <cstdio>

typedef float matrix4[4][4];
typedef float vector3[3];

// check for OpenGL errors
inline
void checkGLErrors(const char *s)
{
    EGLenum error;

    while ((error = glGetError()) != GL_NO_ERROR)
    {
        fprintf(stderr, "%s: error - %d\n", s, error);
    }
}

class ParticleRenderer
{
    public:
        ParticleRenderer(unsigned int windowWidth = 720, unsigned int windowHeight = 480);
        ~ParticleRenderer();

        void setPositions(float *pos, int numParticles);
        void setPositions(double *pos, int numParticles);
        void setBaseColor(float color[4]);
        void setColors(float *color, int numParticles);
        void setPBO(unsigned int pbo, int numParticles, bool fp64);

        enum DisplayMode
        {
            PARTICLE_POINTS,
            PARTICLE_SPRITES,
            PARTICLE_SPRITES_COLOR,
            PARTICLE_NUM_MODES
        };

        void display();

        void setPointSize(float size)
        {
            m_pointSize = size;
        }
        void setSpriteSize(float size)
        {
            m_spriteSize = size;
        }
        
        void setCameraPos(vector3 camera_pos)
        {
            m_camera[0] = camera_pos[0];
            m_camera[1] = camera_pos[1];
            m_camera[2] = camera_pos[2];
        }

        void resetPBO();

    protected: // methods
        void _initGL();
        void _createTexture(int resolution);


    protected: // data
        float *m_pos;
        double *m_pos_fp64;
        int m_numParticles;

        float m_pointSize;
        float m_spriteSize;
        vector3 m_camera;

        unsigned int m_vertexShader;
        unsigned int m_vertexShaderPoints;
        unsigned int m_fragmentShader;
        unsigned int m_programPoints;
        unsigned int m_programSprites;
        unsigned int m_texture;
        unsigned int m_pbo;
        unsigned int m_vboColor;
        unsigned int m_windowWidth;
        unsigned int m_windowHeight;

        float m_baseColor[4];

        bool m_bFp64Positions;
};

#endif //__ RENDER_PARTICLES__
