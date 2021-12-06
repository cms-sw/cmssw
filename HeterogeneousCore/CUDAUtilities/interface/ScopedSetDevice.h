#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    class ScopedSetDevice {
    public:
      // Store the original device, without setting a new one
      ScopedSetDevice() {
        // Store the original device
        cudaCheck(cudaGetDevice(&originalDevice_));
      }

      // Store the original device, and set a new current device
      explicit ScopedSetDevice(int device) : ScopedSetDevice() {
        // Change the current device
        set(device);
      }

      // Restore the original device
      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        cudaSetDevice(originalDevice_);
      }

      // Set a new current device, without changing the original device
      // that will be restored when this object is destroyed
      void set(int device) {
        // Change the current device
        cudaCheck(cudaSetDevice(device));
      }

    private:
      int originalDevice_;
    };
  }  // namespace cuda
}  // namespace cms

#endif
