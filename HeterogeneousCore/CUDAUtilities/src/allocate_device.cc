#include <limits>

#include "FWCore/Utilities/interface/Likely.h"
#include "HeterogeneousCore/CUDAUtilities/interface/ScopedSetDevice.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (UNLIKELY(nbytes > maxAllocationSize)) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
