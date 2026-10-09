#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"

#include <utility>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FixedQueueRegistry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/getDeviceCachingAllocator.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <ATen/cuda/CUDABlas.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define USE_ROCM 1  // some PyTorch's HIP headers still use USE_ROCM to select ROCm-specific code paths.
#include <ATen/hip/HIPBlas.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  bool PyTorchAllocatorBridge::active_ = false;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

  void* PyTorchAllocatorBridge::allocate(size_t size, int deviceId, QueueHandle stream) {
    auto queue = cms::alpakatools::getFixedQueueRegistry<Queue>().findQueue(deviceId, stream);

    if (!queue) {
      throw cms::Exception("PyTorchAllocatorBridge")
          << "Could not find an Alpaka Queue associated with device " << deviceId << " and stream " << stream;
    }

    auto const device = alpaka::getDev(*queue);
    auto& allocator = cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);

    return allocator.allocate(size, std::move(*queue));
  }

  void PyTorchAllocatorBridge::free(void* ptr, size_t /*size*/, int deviceId, QueueHandle /*stream*/) {
    if (ptr == nullptr) {
      return;
    }

    auto const& deviceList = cms::alpakatools::devices<Platform>();

    if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceList.size()) {
      throw cms::Exception("PyTorchAllocatorBridge") << "Invalid device ID " << deviceId;
    }

    auto const& device = deviceList[deviceId];
    auto& allocator = cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);

    allocator.free(ptr);
  }

#endif

  void PyTorchAllocatorBridge::install() {
    if (active_)
      return;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    namespace PA = ::torch::cuda::CUDAPluggableAllocator;

    auto allocator = PA::createCustomAllocator(&PyTorchAllocatorBridge::allocate, &PyTorchAllocatorBridge::free);

    PA::changeCurrentAllocator(allocator);
    active_ = true;
#endif
  }

  void PyTorchAllocatorBridge::clearBlasWorkspace(Queue queue) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto stream = alpaka::getNativeHandle(queue);
    at::cuda::clearCublasWorkspacesForStream(stream);
#endif
  }

  bool PyTorchAllocatorBridge::isActive() { return active_; }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
