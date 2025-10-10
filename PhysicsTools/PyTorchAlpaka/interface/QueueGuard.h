#ifndef PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h
#define PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h

#include <type_traits>

#include <alpaka/alpaka.hpp>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#elif ALPAKA_ACC_GPU_HIP_ENABLED
// #include <c10/hip/HIPStream.h>
#endif

#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"

namespace cms::torch::alpakatools {

  // Default no-op implementation for platforms where no special handling is needed.
  // CPU backends (do not need extra handling - multithreading is disabled by `PyTorchService`):
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  // GPU backends:
  // - ALPAKA_ACC_GPU_HIP_ENABLED (AMD ROCm/HIP backend not yet supported, see below)
  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  struct QueueGuardTraits {
    static void set(const TQueue &) noexcept { /* no-op default, threading disabled by `PyTorchService` */ }
    static void reset() noexcept { /* no-op default, once threading is disabled cannot be reset */ }
  };

  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  class QueueGuard {
  public:
    explicit QueueGuard(const TQueue &queue) { QueueGuardTraits<TQueue>::set(queue); }
    ~QueueGuard() { QueueGuardTraits<TQueue>::reset(); }
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  template <>
  struct QueueGuardTraits<alpaka_cuda_async::Queue> {
    inline static thread_local c10::cuda::CUDAStream cached_stream_ = c10::cuda::getCurrentCUDAStream();
    // setCurrentCUDAStream() is assumed to not throw exceptions on the later-than-first calls.
    // see: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L373
    // Internal torch implementation of CUDA stream handling is based on a `thread_local`
    // see: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L169
    // follows the semantics of "current device" of CUDA itself (but not of Alpaka)
    //
    // TODO: `noexcept` is used to avoid exceptions in the destructor, which for 100% clarity
    // restore the previous state. Not required for correctness and could be neglected due to override behavior.
    static void set(const alpaka_cuda_async::Queue &queue) noexcept {
      auto dev = ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue);
      auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
      cached_stream_ = c10::cuda::getCurrentCUDAStream();
      c10::cuda::setCurrentCUDAStream(stream);
    }
    static void reset() noexcept { c10::cuda::setCurrentCUDAStream(cached_stream_); }
  };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

  // AMD ROCm/HIP backend not yet supported (though Alpaka HIP backend is available), the CPU fallback is used.
  // When CMSSW provide `pytorch-hip` counterpart in addition to `pytorch-cuda`, this can be implemented analogously to CUDA above.
  // See:
  // - https://docs.pytorch.org/docs/stable/notes/hip.html
  // - https://github.com/pytorch/pytorch/tree/v2.6.0/c10/cuda#readme c10::cuda -> c10::hip
  //
  //
  //
  // template <>
  // struct QueueGuardTraits<alpaka_rocm_async::Queue> {
  //   static void set(const alpaka_rocm_async::Queue &queue) noexcept {
  //     auto dev = ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue);
  //     auto stream = c10::hip::getStreamFromExternal(queue.getNativeHandle(), dev.index());
  //     c10::hip::setCurrentCUDAStream(stream);
  //   }
  //   static void reset() noexcept { /* no-op */ }
  // };

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // __________________________________________________________________________________________________________________
  // Utilities for debugging queue managament

  // Default fallback for CPU / unknown
  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  struct QueueHash {
    static std::string alpakaQueue(const TQueue &) {
      return fmt::format("{:#x}", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }
    static std::string pytorchQueue(const TQueue &queue) { return alpakaQueue(queue); }
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  struct QueueHash<alpaka_cuda_async::Queue> {
    static std::string alpakaQueue(const alpaka_cuda_async::Queue &queue) {
      auto cuStream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(),
                                                       ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue).index());
      unsigned long long streamId{};
      cudaStreamGetId(cuStream, &streamId);
      return fmt::format("{:#d}", streamId);
    }
    static std::string pytorchQueue(const alpaka_cuda_async::Queue &queue) {
      auto stream = c10::cuda::getCurrentCUDAStream(ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue).index());
      unsigned long long streamId{};
      cudaStreamGetId(stream, &streamId);
      return fmt::format("{:#d}", streamId);
    }
  };
#endif

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h