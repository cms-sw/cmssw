#ifndef HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h

#include <cassert>
#include <memory>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatools {

  namespace detail {

    template <typename TDev,
              typename TQueue,
              typename = std::enable_if_t<alpaka::isDevice<TDev> and alpaka::isQueue<TQueue>>>
    auto allocate_device_allocators() {
      using Allocator = CachingAllocator<TDev, TQueue>;
      auto const& devices = cms::alpakatools::devices<alpaka::Pltf<TDev>>();
      ssize_t const size = devices.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      ptrdiff_t index = 0;
      try {
        for (; index < size; ++index) {
#if __cplusplus >= 202002L
          std::construct_at(
#else
          std::allocator<Allocator>().construct(
#endif
              ptr + index,
              devices[index],
              config::binGrowth,
              config::minBin,
              config::maxBin,
              config::maxCachedBytes,
              config::maxCachedFraction,
              true,    // reuseSameQueueAllocations
              false);  // debug
        }
      } catch (...) {
        --index;
        // destroy any object that had been succesfully constructed
        while (index >= 0) {
          std::destroy_at(ptr + index);
          --index;
        }
        // deallocate the storage
        std::allocator<Allocator>().deallocate(ptr, size);
        // rethrow the exception
        throw;
      }

      // use a custom deleter to destroy all objects and deallocate the memory
      auto deleter = [size](Allocator* ptr) {
        for (size_t i = size; i > 0; --i) {
          std::destroy_at(ptr + i - 1);
        }
        std::allocator<Allocator>().deallocate(ptr, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  template <typename TDev,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isDevice<TDev> and alpaka::isQueue<TQueue>>>
  inline CachingAllocator<TDev, TQueue>& getDeviceCachingAllocator(TDev const& device) {
    // initialise all allocators, one per device
    CMS_THREAD_SAFE static auto allocators = detail::allocate_device_allocators<TDev, TQueue>();

    size_t const index = alpaka::getNativeHandle(device);
    assert(index < cms::alpakatools::devices<alpaka::Pltf<TDev>>().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h
