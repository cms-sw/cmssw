#ifndef HeterogenousCore_CUDAUtilities_cudaDeviceCount_h
#define HeterogenousCore_CUDAUtilities_cudaDeviceCount_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cudautils {
  inline int cudaDeviceCount() {
    int ndevices;
    cudaCheck(cudaGetDeviceCount(&ndevices));
    return ndevices;
  }
}  // namespace cudautils

#endif
