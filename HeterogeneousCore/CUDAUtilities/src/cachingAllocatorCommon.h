#ifndef HeterogeneousCore_CUDACore_src_cachingAllocatorCommon
#define HeterogeneousCore_CUDACore_src_cachingAllocatorCommon

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/deviceCount.h"

#include <algorithm>
#include <limits>

namespace cms::cuda::allocator {
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 2;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 8;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 30;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    int currentDevice;
    cudaCheck(cudaGetDevice(&currentDevice));
    const int numberOfDevices = deviceCount();
    for (int i = 0; i < numberOfDevices; ++i) {
      size_t freeMemory, totalMemory;
      cudaCheck(cudaSetDevice(i));
      cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    cudaCheck(cudaSetDevice(currentDevice));
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }
}  // namespace cms::cuda::allocator

#endif
