#ifndef HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h

#include <cassert>
#include <memory>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

namespace cms::alpakatools {

  namespace detail {

    template <typename TDev,
              typename TQueue,
              typename = std::enable_if_t<alpaka::isDevice<TDev> and alpaka::isQueue<TQueue>>>
    auto allocate_device_allocators(AllocatorConfig const& config, bool debug) {
      using Allocator = CachingAllocator<TDev, TQueue>;
      auto const& devices = cms::alpakatools::devices<alpaka::Platform<TDev>>();
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
              config,
              true,  // reuseSameQueueAllocations
              debug);
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
      auto deleter = [size](Allocator* allocators) {
        for (size_t i = size; i > 0; --i) {
          std::destroy_at(allocators + i - 1);
        }
        std::allocator<Allocator>().deallocate(allocators, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  template <typename TDev,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isDevice<TDev> and alpaka::isQueue<TQueue>>>
  inline CachingAllocator<TDev, TQueue>& getDeviceCachingAllocator(TDev const& device,
                                                                   AllocatorConfig const& config = AllocatorConfig{},
                                                                   bool debug = false) {
    // initialise all allocators, one per device
    CMS_THREAD_SAFE static auto allocators = detail::allocate_device_allocators<TDev, TQueue>(config, debug);

    size_t const index = alpaka::getNativeHandle(device);
    assert(index < cms::alpakatools::devices<alpaka::Platform<TDev>>().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_getDeviceCachingAllocator_h
