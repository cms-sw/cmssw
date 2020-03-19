#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/deviceCount.h"
#include "CachingDeviceAllocator.h"

#include <iomanip>

namespace cms::cuda::allocator {
  // Use caching or not
  constexpr bool useCaching = true;
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 8;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 1;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 10;
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

  inline notcub::CachingDeviceAllocator& getCachingDeviceAllocator() {
    LogDebug("CachingDeviceAllocator").log([](auto& log) {
      log << "cub::CachingDeviceAllocator settings\n"
          << "  bin growth " << binGrowth << "\n"
          << "  min bin    " << minBin << "\n"
          << "  max bin    " << maxBin << "\n"
          << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = notcub::CachingDeviceAllocator::IntPow(binGrowth, bin);
        if (binSize >= (1 << 30) and binSize % (1 << 30) == 0) {
          log << "    " << std::setw(8) << (binSize >> 30) << " GB\n";
        } else if (binSize >= (1 << 20) and binSize % (1 << 20) == 0) {
          log << "    " << std::setw(8) << (binSize >> 20) << " MB\n";
        } else if (binSize >= (1 << 10) and binSize % (1 << 10) == 0) {
          log << "    " << std::setw(8) << (binSize >> 10) << " kB\n";
        } else {
          log << "    " << std::setw(9) << binSize << " B\n";
        }
      }
      log << "  maximum amount of cached memory: " << (minCachedBytes() >> 20) << " MB\n";
    });

    // the public interface is thread safe
    CMS_THREAD_SAFE static notcub::CachingDeviceAllocator allocator{binGrowth,
                                                                    minBin,
                                                                    maxBin,
                                                                    minCachedBytes(),
                                                                    false,  // do not skip cleanup
                                                                    debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
