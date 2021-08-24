#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cachingAllocators.h"

#include "CachingDeviceAllocator.h"
#include "cachingAllocatorCommon.h"

#include <iomanip>

namespace cms::cuda::allocator {
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
