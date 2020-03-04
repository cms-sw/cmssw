#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
        cudaCheck(cudaGetDevice(&prevDevice_));
        cudaCheck(cudaSetDevice(newDevice));
      }

      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        cudaSetDevice(prevDevice_);
      }

    private:
      int prevDevice_;
    };
  }  // namespace cuda
}  // namespace cms

#endif
