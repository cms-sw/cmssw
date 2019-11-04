#ifndef HeterogeneousCore_CUDAUtilities_eventIsOccurred_h
#define HeterogeneousCore_CUDAUtilities_eventIsOccurred_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cudautils {
  inline bool eventIsOccurred(cudaEvent_t event) {
    const auto ret = cudaEventQuery(event);
    if (ret == cudaSuccess) {
      return true;
    } else if (ret == cudaErrorNotReady) {
      return false;
    }
    // leave error case handling to cudaCheck
    cudaCheck(ret);
    return false;  // to keep compiler happy
  }
}  // namespace cudautils

#endif
