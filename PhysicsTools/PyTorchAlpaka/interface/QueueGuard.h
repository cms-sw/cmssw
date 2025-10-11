#ifndef PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h
#define PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h

#include <type_traits>

#include <alpaka/alpaka.hpp>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#elif ALPAKA_ACC_GPU_HIP_ENABLED
// #include <c10/hip/HIPStream.h>
#endif

#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"

namespace cms::torch::alpakatools {

  // Default no-op implementation for platforms where no special handling is needed.
  // CPU backends (do not need extra handling - multithreading is disabled by `PyTorchService`):
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  // GPU backends:
  // - ALPAKA_ACC_GPU_HIP_ENABLED (AMD ROCm/HIP backend not yet supported, see below)
  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  class QueueGuard {
  public:
    explicit QueueGuard(const TQueue &queue) { /* no-op default, threading disabled by `PyTorchService` */ }
    ~QueueGuard() noexcept { /* no-op default, once threading is disabled cannot be reset */ }
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  template <>
  class QueueGuard<alpaka_cuda_async::Queue> {
  public:
    explicit QueueGuard(const alpaka_cuda_async::Queue &queue) noexcept
        : cached_stream_{c10::cuda::getCurrentCUDAStream()} {
      auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), getDevice(queue).index());
      c10::cuda::setCurrentCUDAStream(stream);
    }

    ~QueueGuard() noexcept { c10::cuda::setCurrentCUDAStream(cached_stream_); }

  private:
    c10::cuda::CUDAStream cached_stream_ = c10::cuda::getCurrentCUDAStream();
  };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

  // AMD ROCm/HIP backend not yet supported (though Alpaka HIP backend is available), the CPU fallback is used.
  // When CMSSW provide `pytorch-hip` counterpart in addition to `pytorch-cuda`, this can be implemented analogously to CUDA above.
  // See:
  // - https://docs.pytorch.org/docs/stable/notes/hip.html
  // - https://github.com/pytorch/pytorch/tree/v2.6.0/c10/cuda#readme c10::cuda -> c10::hip (fn syntax may keep CUDA)
  //
  // template <>
  // class QueueGuard<alpaka_rocm_async::Queue> {
  // public:
  //   explicit QueueGuard(const alpaka_rocm_async::Queue &queue) noexcept
  //       : cached_stream_{c10::hip::getCurrentHIPStream()} {
  //     auto stream = c10::hip::getStreamFromExternal(queue.getNativeHandle(), getDevice(queue).index());
  //     c10::hip::setCurrentHIPStream(stream);
  //   }
  //   ~QueueGuard() noexcept {
  //     // Restore the previous HIP stream
  //     c10::hip::setCurrentHIPStream(cached_stream_);
  //   }
  // private:
  //   c10::hip::HIPStream cached_stream_ = c10::cuda::getCurrentCUDAStream();
  // };

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h
