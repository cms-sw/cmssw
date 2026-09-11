#ifndef PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h
#define PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h

#include <type_traits>

#include <alpaka/alpaka.hpp>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <c10/hip/HIPStream.h>
#endif

#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"

namespace cms::torch::alpakatools {

  // Default no-op implementation for platforms where no special handling is needed.
  // CPU backends (do not need extra handling - multithreading is disabled by `PyTorchService`):
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  class QueueGuard {
  public:
    explicit QueueGuard(const TQueue &queue) { /* no-op default, threading disabled by `PyTorchService` */ }
    ~QueueGuard() noexcept { /* no-op default, once threading is disabled cannot be reset */ }
  };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  using AsyncQueue = alpaka_cuda_async::Queue;
#else
  using AsyncQueue = alpaka_rocm_async::Queue;
#endif

  template <>
  class QueueGuard<AsyncQueue> {
  public:
    explicit QueueGuard(const AsyncQueue &queue) noexcept : cached_stream_{c10::cuda::getCurrentCUDAStream()} {
      auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), getDevice(queue).index());
      c10::cuda::setCurrentCUDAStream(stream);
    }

    ~QueueGuard() noexcept { c10::cuda::setCurrentCUDAStream(cached_stream_); }

  private:
    c10::cuda::CUDAStream cached_stream_ = c10::cuda::getCurrentCUDAStream();
  };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h
