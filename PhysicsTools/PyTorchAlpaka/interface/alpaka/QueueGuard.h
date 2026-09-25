#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_QueueGuard_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_QueueGuard_h

#include <alpaka/alpaka.hpp>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <c10/hip/HIPStream.h>
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  class QueueGuard {
  public:
    explicit QueueGuard(const Queue &queue) noexcept : cached_stream_{c10::cuda::getCurrentCUDAStream()} {
      auto stream =
          c10::cuda::getStreamFromExternal(queue.getNativeHandle(), cms::torch::alpakatools::getDevice(queue).index());
      c10::cuda::setCurrentCUDAStream(stream);
    }

    ~QueueGuard() noexcept { c10::cuda::setCurrentCUDAStream(cached_stream_); }

  private:
    c10::cuda::CUDAStream cached_stream_;
  };
#else
  class QueueGuard {
  public:
    explicit QueueGuard(const Queue &) noexcept {}
    ~QueueGuard() = default;
  };

#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_QueueGuard_h
