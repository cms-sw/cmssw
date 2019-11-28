#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cudautils {
  inline int currentDevice() {
    int dev;
    cudaCheck(cudaGetDevice(&dev));
    return dev;
  }
}  // namespace cudautils

#endif
