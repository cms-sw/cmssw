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
 * This sample demonstrates how to use query information on the current system
 * topology using a SDK 8.0 API.
 */

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

int main(int argc, char **argv)
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Enumerates Device <-> Device links
    for (int device1 = 0; device1 < deviceCount; device1++)
    {
        for (int device2 = 0; device2 < deviceCount; device2++)
        {
            if (device1 == device2)
                continue;

            int perfRank = 0;
            int atomicSupported = 0;
            int accessSupported = 0;

            checkCudaErrors(cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2));
            checkCudaErrors(cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2));
            checkCudaErrors(cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2));

            if (accessSupported)
            {
                std::cout << "GPU" << device1 << " <-> GPU" << device2 << ":" << std::endl;
                std::cout << "  * Atomic Supported: " << (atomicSupported ? "yes" : "no") << std::endl;
                std::cout << "  * Perf Rank: " << perfRank << std::endl;
            }
        }
    }

    // Enumerates Device <-> Host links
    for (int device = 0; device < deviceCount; device++)
    {
        int atomicSupported = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, device));
        std::cout << "GPU" << device << " <-> CPU:" << std::endl;
        std::cout << "  * Atomic Supported: " << (atomicSupported ? "yes" : "no") << std::endl;
    }

    return 0;
}
