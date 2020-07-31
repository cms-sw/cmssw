#include "HeterogeneousCore/CUDAUtilities/interface/deviceAllocatorStatus.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda {
  allocator::GpuCachedBytes deviceAllocatorStatus() { return allocator::getCachingDeviceAllocator().CacheStatus(); }
}  // namespace cms::cuda
