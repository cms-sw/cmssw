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

// This sample is an implementation of a simple line-of-sight algorithm:
// Given a height map and a ray originating at some observation point,
// it computes all the points along the ray that are visible from the
// observation point.
// It is based on the description made in "Guy E. Blelloch.  Vector models
// for data-parallel computing. MIT Press, 1990" and uses open source CUDA
// Thrust Library

#ifdef _WIN32
#  define NOMINMAX
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>

// includes, library
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, types

// Boolean
typedef unsigned char Bool;
enum
{
    False = 0,
    True = 1
};

// 2D height field
struct HeightField
{
    int     width;
    float  *height;
};

// Ray
struct Ray
{
    float3 origin;
    float2 dir;
    int    length;
    float  oneOverLength;
};

////////////////////////////////////////////////////////////////////////////////
// declaration, variables

// Height field texture reference
texture<float, 2, cudaReadModeElementType> g_HeightFieldTex;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, char **argv);
__global__ void computeAngles_kernel(const Ray, float *);
__global__ void computeVisibilities_kernel(const float *, const float *, int, Bool *);
void lineOfSight_gold(const HeightField, const Ray, Bool *);
__device__ __host__ float2 getLocation(const Ray, int);
__device__ __host__ float getAngle(const Ray, float2, float);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    int res = runTest(argc, argv);

    if (res != 1)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);

}

////////////////////////////////////////////////////////////////////////////////
//! Run a line-of-sight test for CUDA
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, char **argv)
{
    ////////////////////////////////////////////////////////////////////////////
    // Device initialization

    printf("[%s] - Starting...\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    ////////////////////////////////////////////////////////////////////////////
    // Timer

    // Create
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // Number of iterations to get accurate timing
    uint numIterations = 100;

    ////////////////////////////////////////////////////////////////////////////
    // Height field

    HeightField heightField;

    // Allocate in host memory
    int2 dim = make_int2(10000, 100);
    heightField.width = dim.x;
    thrust::host_vector<float> height(dim.x * dim.y);
    heightField.height = (float *)&height[0];

    //
    // Fill in with an arbitrary sine surface
    for (int x = 0; x < dim.x; ++x)
        for (int y = 0; y < dim.y; ++y)
        {
            float amp = 0.1f * (x + y);
            float period = 2.0f + amp;
            *(heightField.height + dim.x * y + x) =
                amp * (sinf(sqrtf((float)(x * x + y * y)) * 2.0f * 3.1416f / period) + 1.0f);
        }

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *heightFieldArray;
    checkCudaErrors(cudaMallocArray(&heightFieldArray, &channelDesc, dim.x, dim.y));

    // Initialize device memory
    checkCudaErrors(cudaMemcpyToArray(heightFieldArray, 0, 0, heightField.height,
                                      dim.x * dim.y * sizeof(float), cudaMemcpyHostToDevice));

    // Set texture parameters
    g_HeightFieldTex.addressMode[0] = cudaAddressModeClamp;
    g_HeightFieldTex.addressMode[1] = cudaAddressModeClamp;
    g_HeightFieldTex.filterMode = cudaFilterModePoint;
    g_HeightFieldTex.normalized = 0;

    // Bind CUDA array to texture reference
    checkCudaErrors(cudaBindTextureToArray(g_HeightFieldTex, heightFieldArray,
                                           channelDesc));

    ////////////////////////////////////////////////////////////////////////////
    // Ray (starts at origin and traverses the height field diagonally)

    Ray ray;
    ray.origin = make_float3(0, 0, 2.0f);
    int2 dir = make_int2(dim.x - 1, dim.y - 1);
    ray.dir = make_float2((float)dir.x, (float)dir.y);
    ray.length = max(abs(dir.x), abs(dir.y));
    ray.oneOverLength = 1.0f / ray.length;

    ////////////////////////////////////////////////////////////////////////////
    // View angles

    // Allocate view angles for each point along the ray
    thrust::device_vector<float> d_angles(ray.length);

    // Allocate result of max-scan operation on the array of view angles
    thrust::device_vector<float> d_scannedAngles(ray.length);

    ////////////////////////////////////////////////////////////////////////////
    // Visibility results

    // Allocate visibility results for each point along the ray
    thrust::device_vector<Bool> d_visibilities(ray.length);
    thrust::host_vector<Bool> h_visibilities(ray.length);
    thrust::host_vector<Bool> h_visibilitiesRef(ray.length);

    ////////////////////////////////////////////////////////////////////////////
    // Reference solution
    lineOfSight_gold(heightField, ray, (Bool *)&h_visibilitiesRef[0]);

    ////////////////////////////////////////////////////////////////////////////
    // Device solution

    // Execution configuration
    dim3 block(256);
    dim3 grid((uint)ceil(ray.length / (double)block.x));

    // Compute device solution
    printf("Line of sight\n");
    sdkStartTimer(&timer);

    for (uint i = 0; i < numIterations; ++i)
    {

        // Compute view angle for each point along the ray
        computeAngles_kernel<<<grid, block>>>(ray, thrust::raw_pointer_cast(&d_angles[0]));
        getLastCudaError("Kernel execution failed");

        // Perform a max-scan operation on the array of view angles
        thrust::inclusive_scan(d_angles.begin(), d_angles.end(), d_scannedAngles.begin(), thrust::maximum<float>());
        getLastCudaError("Kernel execution failed");

        // Compute visibility results based on the array of view angles
        // and its scanned version
        computeVisibilities_kernel<<<grid, block>>>(thrust::raw_pointer_cast(&d_angles[0]),
                                                    thrust::raw_pointer_cast(&d_scannedAngles[0]),
                                                    ray.length,
                                                    thrust::raw_pointer_cast(&d_visibilities[0]));
        getLastCudaError("Kernel execution failed");
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    getLastCudaError("Kernel execution failed");

    // Copy visibility results back to the host
    thrust::copy(d_visibilities.begin(), d_visibilities.end(), h_visibilities.begin());

    // Compare device visibility results against reference results
    bool res = compareData(thrust::raw_pointer_cast(&h_visibilitiesRef[0]),
                           thrust::raw_pointer_cast(&h_visibilities[0]), ray.length, 0.0f, 0.0f);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&timer) / numIterations);
    sdkResetTimer(&timer);

    // Cleanup memory
    checkCudaErrors(cudaFreeArray(heightFieldArray));
    return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute view angles for each point along the ray
//! @param ray         ray
//! @param angles      view angles
////////////////////////////////////////////////////////////////////////////////
__global__ void computeAngles_kernel(const Ray ray, float *angles)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < ray.length)
    {
        float2 location = getLocation(ray, i + 1);
        float height = tex2D(g_HeightFieldTex, location.x, location.y);
        float angle = getAngle(ray, location, height);
        angles[i] = angle;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute visibility for each point along the ray
//! @param angles          view angles
//! @param scannedAngles   max-scanned view angles
//! @param numAngles       number of view angles
//! @param visibilities    boolean array indicating the visibility of each point
//!                        along the ray
////////////////////////////////////////////////////////////////////////////////
__global__ void computeVisibilities_kernel(const float *angles,
                                           const float *scannedAngles,
                                           int numAngles,
                                           Bool *visibilities)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numAngles)
    {
        visibilities[i] = scannedAngles[i] <= angles[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param heightField     height field
//! @param ray             ray
//! @param visibilities    boolean array indicating the visibility of each point
//!                        along the ray
////////////////////////////////////////////////////////////////////////////////
void lineOfSight_gold(const HeightField heightField, const Ray ray,
                      Bool *visibilities)
{
    float angleMax = asinf(-1.0f);

    for (int i = 0; i < ray.length; ++i)
    {
        float2 location = getLocation(ray, i + 1);
        float height = *(heightField.height
                         + heightField.width * (int)floorf(location.y)
                         + (int)floorf(location.x));
        float angle = getAngle(ray, location, height);

        if (angle > angleMax)
        {
            angleMax = angle;
            visibilities[i] = True;
        }
        else
        {
            visibilities[i] = False;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the 2D coordinates of the point located at i steps from the origin
//! of the ray
//! @param ray      ray
//! @param i        integer offset along the ray
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ float2 getLocation(const Ray ray, int i)
{
    float step = i * ray.oneOverLength;
    return make_float2(ray.origin.x, ray.origin.y) + ray.dir * step;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the angle of view between a 3D point and the origin of the ray
//! @param ray        ray
//! @param location   2D coordinates of the input point
//! @param height     height of the input point
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ float getAngle(const Ray ray, float2 location, float height)
{
    float2 dir = location - make_float2(ray.origin.x, ray.origin.y);
    return atanf((height - ray.origin.z) / length(dir));
}
