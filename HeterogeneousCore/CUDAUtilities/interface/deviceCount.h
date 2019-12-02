#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    inline int deviceCount() {
      int ndevices;
      cudaCheck(cudaGetDeviceCount(&ndevices));
      return ndevices;
    }
  }  // namespace cuda
}  // namespace cms

#endif
