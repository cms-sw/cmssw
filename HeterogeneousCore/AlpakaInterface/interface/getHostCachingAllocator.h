#ifndef HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatools {

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  inline CachingAllocator<alpaka_common::DevHost, TQueue>& getHostCachingAllocator() {
    // thread safe initialisation of the host allocator
    CMS_THREAD_SAFE static CachingAllocator<alpaka_common::DevHost, TQueue> allocator(
        host(),
        config::binGrowth,
        config::minBin,
        config::maxBin,
        config::maxCachedBytes,
        config::maxCachedFraction,
        false,   // reuseSameQueueAllocations
        false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h
