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

attribute vec4 a_position;

uniform mat4 projection;
uniform mat4 modelview;

void main()
{
    gl_PointSize = 1.0;
    gl_Position = projection * modelview * a_position;
}

