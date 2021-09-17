#ifndef HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of cudaEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(cudaEvent_t event) {
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
  }  // namespace cuda
}  // namespace cms

#endif
