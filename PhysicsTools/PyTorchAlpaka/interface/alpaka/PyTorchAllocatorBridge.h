#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAllocatorBridge_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAllocatorBridge_h

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <utility>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // PyTorch provides the device and native stream for each allocation.
  // The PyTorchAllocatorBridge makes use of the FixedQueueRegistry to retrieve the associated Alpaka queue
  // and call the CMSSW caching allocator functions.
  class PyTorchAllocatorBridge {
  public:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using QueueHandle = decltype(alpaka::getNativeHandle(std::declval<Queue>()));

    static void* allocate(size_t size, int deviceId, QueueHandle stream);
    static void free(void* ptr, size_t size, int deviceId, QueueHandle /*stream*/);
#endif
    static void install();
    static void clearBlasWorkspace(Queue queue);
    static bool isActive();

  private:
    static bool active_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
