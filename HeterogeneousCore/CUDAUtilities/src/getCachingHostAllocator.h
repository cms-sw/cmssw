#ifndef HeterogeneousCore_CUDACore_src_getCachingHostAllocator
#define HeterogeneousCore_CUDACore_src_getCachingHostAllocator

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CachingHostAllocator.h"

#include "getCachingDeviceAllocator.h"

#include <iomanip>

namespace cudautils {
  namespace allocator {
    inline notcub::CachingHostAllocator& getCachingHostAllocator() {
      LogDebug("CachingHostAllocator").log([](auto& log) {
        log << "cub::CachingHostAllocator settings\n"
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

      static notcub::CachingHostAllocator allocator{binGrowth,
                                                    minBin,
                                                    maxBin,
                                                    minCachedBytes(),
                                                    false,  // do not skip cleanup
                                                    debug};
      return allocator;
    }
  }  // namespace allocator
}  // namespace cudautils

#endif
