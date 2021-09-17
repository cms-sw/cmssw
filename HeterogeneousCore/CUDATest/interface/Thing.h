#ifndef HeterogeneousCore_CUDATest_Thing_H
#define HeterogeneousCore_CUDATest_Thing_H

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

namespace cms {
  namespace cudatest {
    class Thing {
    public:
      Thing() = default;
      explicit Thing(cms::cuda::device::unique_ptr<float[]> ptr) : ptr_(std::move(ptr)) {}

      const float *get() const { return ptr_.get(); }

    private:
      cms::cuda::device::unique_ptr<float[]> ptr_;
    };
  }  // namespace cudatest
}  // namespace cms

#endif
