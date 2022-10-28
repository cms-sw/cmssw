#include "HeterogeneousCore/CUDAUtilities/interface/cachingAllocators.h"

#include "getCachingDeviceAllocator.h"
#include "getCachingHostAllocator.h"

namespace cms::cuda::allocator {
  void cachingAllocatorsConstruct() {
    cms::cuda::allocator::getCachingDeviceAllocator();
    cms::cuda::allocator::getCachingHostAllocator();
  }

  void cachingAllocatorsFreeCached() {
    cms::cuda::allocator::getCachingDeviceAllocator().FreeAllCached();
    cms::cuda::allocator::getCachingHostAllocator().FreeAllCached();
  }
}  // namespace cms::cuda::allocator
