#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include <cstddef>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::torch::alpakatools {

  // Default no-ops policy for fully supported backends:
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_GPU_CUDA_ENABLED
  struct DefaultPolicy {
    explicit DefaultPolicy(const void*, size_t) {}

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToHost(TQueue&) const noexcept {}

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToDevice(TQueue&) const noexcept {}
  };

  // Generic fallback mechanism that provides a host-resident mirror of device memory blob.
  // Manages a host-side buffer that mirrors device memory, enabling CPU-based inference
  // when ROCm HIP execution is not available.
  struct HipPolicy {
    HipPolicy(const void* d_ptr, const size_t nbytes)
        : d_ptr_(d_ptr), h_buf_(cms::alpakatools::make_host_buffer<std::byte[]>(nbytes)) {}

    // Synchronization (if applicable) responsibility move to the caller
    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToHost(TQueue& queue) {
      auto ptr = const_cast<std::byte*>(reinterpret_cast<const std::byte*>(d_ptr_));
      auto d_view = alpaka::createView(alpaka::getDev(queue), ptr, alpaka::getExtents(h_buf_)[0]);
      alpaka::memcpy(queue, h_buf_, d_view);
    }

    // Synchronization (if applicable) responsibility move to the caller
    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToDevice(TQueue& queue) {
      auto ptr = const_cast<std::byte*>(reinterpret_cast<const std::byte*>(d_ptr_));
      auto d_view = alpaka::createView(alpaka::getDev(queue), ptr, alpaka::getExtents(h_buf_)[0]);
      alpaka::memcpy(queue, d_view, h_buf_);
    }

    const void* hostPtr() const noexcept { return alpaka::getPtrNative(h_buf_); }

  private:
    const void* d_ptr_;
    cms::alpakatools::host_buffer<std::byte[]> h_buf_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h
