#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "FWCore/Utilities/interface/Likely.h"

#include "getCachingHostAllocator.h"

#include <cuda/api_wrappers.h>

#include <limits>

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cudautils::allocator::binGrowth, cudautils::allocator::maxBin);
}

namespace cudautils {
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (cudautils::allocator::useCaching) {
      if (UNLIKELY(nbytes > maxAllocationSize)) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cuda::throw_if_error(cudautils::allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    } else {
      cuda::throw_if_error(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

  void free_host(void *ptr) {
    if constexpr (cudautils::allocator::useCaching) {
      cuda::throw_if_error(cudautils::allocator::getCachingHostAllocator().HostFree(ptr));
    } else {
      cuda::throw_if_error(cudaFreeHost(ptr));
    }
  }

}  // namespace cudautils
