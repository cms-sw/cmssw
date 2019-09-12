#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "FWCore/Utilities/interface/Likely.h"

#include "getCachingDeviceAllocator.h"

#include <limits>

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cudautils::allocator::binGrowth, cudautils::allocator::maxBin);
}

namespace cudautils {
  void *allocate_device(int dev, size_t nbytes, cuda::stream_t<> &stream) {
    void *ptr = nullptr;
    if constexpr (cudautils::allocator::useCaching) {
      if (UNLIKELY(nbytes > maxAllocationSize)) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cuda::throw_if_error(
          cudautils::allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream.id()));
    } else {
      cuda::device::current::scoped_override_t<> setDeviceForThisScope(dev);
      cuda::throw_if_error(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(int device, void *ptr) {
    if constexpr (cudautils::allocator::useCaching) {
      cuda::throw_if_error(cudautils::allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
    } else {
      cuda::device::current::scoped_override_t<> setDeviceForThisScope(device);
      cuda::throw_if_error(cudaFree(ptr));
    }
  }

}  // namespace cudautils
