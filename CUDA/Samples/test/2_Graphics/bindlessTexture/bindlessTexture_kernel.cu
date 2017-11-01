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

/*
    This sample has two kernels, one doing the rendering every frame, and another
    one used to generate the mip map levels at startup.

    For rendering we use a "virtual" texturing approach, where one 2d texture
    stores pointers to the actual textures used. This can be achieved by the
    new cudaTextureObject introduced in CUDA 5.0 and requiring sm3+ hardware.

    The mipmap generation kernel uses cudaSurfaceObject and cudaTextureObject
    passed as kernel arguments to compute the higher mip map level based on
    the lower.

*/

#ifndef _BINDLESSTEXTURE_KERNEL_CU_
#define _BINDLESSTEXTURE_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <vector>

#include <helper_cuda.h>
#include <helper_math.h>

#include "bindlessTexture.h"

// set this to just see the mipmap chain of first image
//#define SHOW_MIPMAPS


// local references to resources

Image               atlasImage;
std::vector<Image>  contentImages;
float               highestLod = 1.0f;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

//////////////////////////////////////////////////////////////////////////

__host__ __device__ __inline__ uint2 encodeTextureObject(cudaTextureObject_t obj)
{
    return make_uint2((uint)(obj & 0xFFFFFFFF), (uint)(obj >> 32));
}

__host__ __device__ __inline__ cudaTextureObject_t decodeTextureObject(uint2 obj)
{
    return (((cudaTextureObject_t)obj.x) | ((cudaTextureObject_t)obj.y) << 32);
}

__device__ __inline__ float4 to_float4(uchar4 vec)
{
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

//////////////////////////////////////////////////////////////////////////
// Rendering

texture<uint2, 2, cudaReadModeElementType> atlasTexture;
// the atlas texture stores the 64 bit cudaTextureObjects
// we use it for "virtual" texturing

__global__ void
d_render(uchar4 *d_output, uint imageW, uint imageH, float lod)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;

    if ((x < imageW) && (y < imageH))
    {
        // read from 2D atlas texture and decode texture object
        uint2 texCoded = tex2D(atlasTexture, u, v);
        cudaTextureObject_t tex = decodeTextureObject(texCoded);

        // read from cuda texture object, use template to specify what data will be
        // returned. tex2DLod allows us to pass the lod (mip map level) directly.
        // There is other functions with CUDA 5, e.g. tex2DGrad,    that allow you
        // to pass derivatives to perform automatic mipmap/anisotropic filtering.
        float4 color = tex2DLod<float4>(tex, u, 1-v, lod);
        // In our sample tex is always valid, but for something like your own
        // sparse texturing you would need to make sure to handle the zero case.

        // write output color
        uint i = y * imageW + x;
        d_output[i] = to_uchar4(color * 255.0);
    }
}

extern "C"
void renderAtlasImage(dim3 gridSize, dim3 blockSize, uchar4 *d_output, uint imageW, uint imageH, float lod)
{
    // psuedo animate lod
    lod = fmodf(lod,highestLod*2);
    lod = highestLod-fabs(lod-highestLod);

#ifdef SHOW_MIPMAPS
    lod = 0.0f;
#endif

    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, lod);

    checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////////////////////
// MipMap Generation


//  A key benefit of using the new surface objects is that we don't need any global
//  binding points anymore. We can directly pass them as function arguments.

__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0/float(imageW);
    float py = 1.0/float(imageH);


    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 1) * py));


        color /= 4.0;
        color *= 255.0;
        color = fminf(color,make_float4(255.0));

        surf2Dwrite(to_uchar4(color),mipOutput,x * sizeof(uchar4),y);
    }
}

void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size)
{
    size_t width    = size.width;
    size_t height   = size.height;

#ifdef SHOW_MIPMAPS
    cudaArray_t levelFirst;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFirst, mipmapArray, 0));
#endif

    uint level = 0;

    while (width != 1 || height != 1)
    {
        width     /= 2;
        width      = MAX((size_t)1,width);
        height    /= 2;
        height     = MAX((size_t)1,height);

        cudaArray_t levelFrom;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo,   mipmapArray, level + 1));

        cudaExtent  levelToSize;
        checkCudaErrors(cudaArrayGetInfo(NULL,&levelToSize,NULL,levelTo));
        checkHost(levelToSize.width  == width);
        checkHost(levelToSize.height == height);
        checkHost(levelToSize.depth  == 0);

        // generate texture object for reading
        cudaTextureObject_t         texInput;
        cudaResourceDesc            texRes;
        memset(&texRes,0,sizeof(cudaResourceDesc));

        texRes.resType            = cudaResourceTypeArray;
        texRes.res.array.array    = levelFrom;

        cudaTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

        // generate surface object for writing

        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc    surfRes;
        memset(&surfRes,0,sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        checkCudaErrors(cudaCreateSurfaceObject(&surfOutput,&surfRes));

        // run mipmap kernel
        dim3 blockSize(16,16,1);
        dim3 gridSize(((uint)width+blockSize.x-1)/blockSize.x,((uint)height+blockSize.y-1)/blockSize.y,1);

        d_mipmap<<<gridSize, blockSize>>>(surfOutput, texInput, (uint)width, (uint)height);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

        checkCudaErrors(cudaDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        cudaMemcpy3DParms copyParams = {0};
        copyParams.dstArray     = levelFirst;
        copyParams.srcArray     = levelTo;
        copyParams.extent       = make_cudaExtent(width,height,1);
        copyParams.kind         = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
#endif

        level++;
    }
}

uint getMipMapLevels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width,size.height),size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}


//////////////////////////////////////////////////////////////////////////
// Initalization

extern "C"
void randomizeAtlas()
{
    uint2 *h_data = (uint2 *) atlasImage.h_data;

    // assign random texture object handles to our atlas image tiles
    for (size_t i = 0; i < atlasImage.size.width * atlasImage.size.height; i++)
    {
#ifdef SHOW_MIPMAPS
        h_data[i] = encodeTextureObject(contentImages[ 0 ].textureObject);
#else
        h_data[i] = encodeTextureObject(contentImages[ rand() % contentImages.size() ].textureObject);
#endif
    }

    // copy data to atlas array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr       = make_cudaPitchedPtr(atlasImage.h_data, atlasImage.size.width * sizeof(uint2), atlasImage.size.width, atlasImage.size.height);
    copyParams.dstArray     = atlasImage.dataArray;
    copyParams.extent       = atlasImage.size;
    copyParams.extent.depth = 1;
    copyParams.kind         = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

};

extern "C"
void deinitAtlasAndImages()
{
    for (size_t i = 0; i < contentImages.size(); i++)
    {
        Image &image = contentImages[i];

        if (image.h_data)
        {
            free(image.h_data);
        }

        if (image.textureObject)
        {
            checkCudaErrors(cudaDestroyTextureObject(image.textureObject));
        }

        if (image.mipmapArray)
        {
            checkCudaErrors(cudaFreeMipmappedArray(image.mipmapArray));
        }
    }

    checkCudaErrors(cudaUnbindTexture(atlasTexture));

    if (atlasImage.h_data)
    {
        free(atlasImage.h_data);
    }

    if (atlasImage.dataArray)
    {
        checkCudaErrors(cudaFreeArray(atlasImage.dataArray));
    }
}

extern "C"
void initAtlasAndImages(const Image *images, size_t numImages, cudaExtent atlasSize)
{
    // create individual textures
    contentImages.resize(numImages);

    for (size_t i = 0; i < numImages; i++)
    {
        Image &image = contentImages[i];
        image.size = images[i].size;
        image.size.depth = 0;
        image.type = cudaResourceTypeMipmappedArray;

        // how many mipmaps we need
        uint levels = getMipMapLevels(image.size);
        highestLod  = MAX(highestLod, (float) levels-1);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        checkCudaErrors(cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, levels));

        // upload level 0
        cudaArray_t level0;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&level0,image.mipmapArray,0));

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr       = make_cudaPitchedPtr(images[i].h_data, image.size.width * sizeof(uchar4), image.size.width, image.size.height);
        copyParams.dstArray     = level0;
        copyParams.extent       = image.size;
        copyParams.extent.depth = 1;
        copyParams.kind         = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // compute rest of mipmaps based on level 0
        generateMipMaps(image.mipmapArray, image.size);

        // generate bindless texture object

        cudaResourceDesc            resDescr;
        memset(&resDescr,0,sizeof(cudaResourceDesc));

        resDescr.resType            = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap  = image.mipmapArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        texDescr.maxMipmapLevelClamp = float(levels - 1);

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));
    }

    // create atlas array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint2>();
    checkCudaErrors(cudaMallocArray(&atlasImage.dataArray, &channelDesc, atlasSize.width, atlasSize.height));
    atlasImage.h_data              = malloc(atlasSize.width * atlasSize.height * sizeof(uint2));
    atlasImage.type                = cudaResourceTypeArray;
    atlasImage.size                = atlasSize;

    // set texture parameters
    atlasTexture.normalized     = 1;                    // access with normalized texture coordinates
    atlasTexture.filterMode     = cudaFilterModePoint;  // and without any filtering (we want raw 2x32bit values)
    atlasTexture.addressMode[0] = cudaAddressModeClamp;
    atlasTexture.addressMode[1] = cudaAddressModeClamp;
    atlasTexture.addressMode[2] = cudaAddressModeClamp;

    randomizeAtlas();

    // Bind array to atlas texture using the classic binding approach.
    // An alternative approach,would be to create a cudaTextureObject and pass it as kernel argument
    checkCudaErrors(cudaBindTextureToArray(atlasTexture, atlasImage.dataArray, channelDesc));
}


#endif // #ifndef _SIMPLETEXTURE3D_KERNEL_CU_
