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

void main()
{
    if(length(gl_PointCoord-vec2(0.5)) > 0.5)
      discard;

    gl_FragColor = vec4(0.0, 1.0, 0.0, 0.5);
}
