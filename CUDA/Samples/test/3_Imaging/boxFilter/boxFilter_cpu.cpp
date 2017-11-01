////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float *id, float *od, int w, int h, int r);

// CPU implementation
void hboxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (2*r+1);

    for (int y = 0; y < h; y++)
    {

        float t;
        // do left edge
        t = id[y*w] * r;

        for (int x = 0; x < r+1; x++)
        {
            t += id[y*w+x];
        }

        od[y*w] = t * scale;

        for (int x = 1; x < r+1; x++)
        {
            int c = y*w+x;
            t += id[c+r];
            t -= id[y*w];
            od[c] = t * scale;
        }

        // main loop
        for (int x = r+1; x < w-r; x++)
        {
            int c = y*w+x;
            t += id[c+r];
            t -= id[c-r-1];
            od[c] = t * scale;
        }

        // do right edge
        for (int x = w-r; x < w; x++)
        {
            int c = y*w+x;
            t += id[(y*w)+w-1];
            t -= id[c-r-1];
            od[c] = t * scale;
        }

    }
}

void hboxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (2*r+1);

    for (int x = 0; x < w; x++)
    {

        float t;
        // do left edge
        t = id[x] * r;

        for (int y = 0; y < r+1; y++)
        {
            t += id[y*w+x];
        }

        od[x] = t * scale;

        for (int y = 1; y < r+1; y++)
        {
            int c = y*w+x;
            t += id[c+r*w];
            t -= id[x];
            od[c] = t * scale;
        }

        // main loop
        for (int y = r+1; y < h-r; y++)
        {
            int c = y*w+x;
            t += id[c+r*w];
            t -= id[c-(r*w)-w];
            od[c] = t * scale;
        }

        // do right edge
        for (int y = h-r; y < h; y++)
        {
            int c = y*w+x;
            t += id[(h-1)*w+x];
            t -= id[c-(r*w)-w];
            od[c] = t * scale;
        }

    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param image      pointer to input data
//! @param temp       pointer to temporary store
//! @param w          width of image
//! @param h          height of image
//! @param r          radius of filter
////////////////////////////////////////////////////////////////////////////////

void computeGold(float *image, float *temp, int w, int h, int r)
{
    hboxfilter_x(image, temp, w, h, r);
    hboxfilter_y(temp, image, w, h, r);
}
