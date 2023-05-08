#ifndef HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h
#define HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h

#include <cstddef>
#include <limits>

namespace cms::alpakatools {

  namespace config {

    // bin growth factor (bin_growth in cub::CachingDeviceAllocator)
    constexpr unsigned int binGrowth = 2;

    // smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CachingDeviceAllocator
    constexpr unsigned int minBin = 8;  // 256 bytes

    // largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
    constexpr unsigned int maxBin = 30;  // 1 GB

    // total storage for the allocator; 0 means no limit.
    constexpr size_t maxCachedBytes = 0;

    // fraction of total device memory taken for the allocator; 0 means no limit.
    constexpr double maxCachedFraction = 0.8;

    // if both maxCachedBytes and maxCachedFraction are non-zero, the smallest resulting value is used.

  }  // namespace config

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h
