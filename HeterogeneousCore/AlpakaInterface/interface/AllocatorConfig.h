#ifndef HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h
#define HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cms::alpakatools {

  struct AllocatorConfig {
    // Bin growth factor (bin_growth in cub::CachingDeviceAllocator)
    unsigned int binGrowth = 2;

    // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CachingDeviceAllocator
    unsigned int minBin = 8;  // 256 bytes

    // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator).
    // Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
    unsigned int maxBin = 30;  // 1 GB

    // Total storage for the allocator; 0 means no limit.
    size_t maxCachedBytes = 0;

    // Fraction of total device memory taken for the allocator; 0 means no limit.
    // If both maxCachedBytes and maxCachedFraction are non-zero, the smallest resulting value is used.
    double maxCachedFraction = 0.8;

    // Fill all newly allocated or re-used memory blocks with fillAllocationValue.
    bool fillAllocations = false;

    // Fill only the re-used memory blocks with fillReallocationValue.
    // If both fillAllocations and fillReallocations are true, fillAllocationValue is used for newly allocated blocks and fillReallocationValue is used for re-allocated blocks.
    bool fillReallocations = false;

    // Fill memory blocks with fillDeallocationValue before freeing or caching them for re-use
    bool fillDeallocations = false;

    // Fill memory blocks with fillCacheValue before caching them for re-use.
    // If both fillDeallocations and fillCaches are true, fillDeallocationValue is used for blocks about to be freed and fillCacheValue is used for blocks about to be cached.
    bool fillCaches = false;

    // Byte value used to fill all newly allocated or re-used memory blocks
    uint8_t fillAllocationValue = 0xA5;

    // Byte value used to fill all re-used memory blocks
    uint8_t fillReallocationValue = 0x69;

    // Byte value used to fill all deallocated or cached memory blocks
    uint8_t fillDeallocationValue = 0x5A;

    // Byte value used to fill all cached memory blocks
    uint8_t fillCacheValue = 0x96;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_AllocatorConfig_h
