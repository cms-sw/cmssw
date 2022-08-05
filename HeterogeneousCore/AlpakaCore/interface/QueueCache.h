#ifndef HeterogeneousCore_AlpakaCore_interface_QueueCache_h
#define HeterogeneousCore_AlpakaCore_interface_QueueCache_h

#include <memory>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace cms::alpakatools {

  template <typename Queue>
  class QueueCache {
    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Pltf<Device>;

  public:
    // QueueCache should be constructed by the first call to
    // getQueueCache() only if we have any devices present
    QueueCache() : cache_(alpaka::getDevCount<Platform>()) {}

    // Gets a (cached) queue for the current device. The queue
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<Queue> get(Device const& dev) {
      return cache_[alpaka::getNativeHandle(dev)].makeOrGet([dev]() { return std::make_unique<Queue>(dev); });
    }

  private:
    // FIXME: not thread safe, intended to be called only from CUDAService destructor ?
    void clear() {
      // Reset the contents of the caches, but leave an
      // edm::ReusableObjectHolder alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // QueueCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_.clear();
      cache_.resize(alpaka::getDevCount<Platform>());
    }

    std::vector<edm::ReusableObjectHolder<Queue>> cache_;
  };

  // Gets the global instance of a QueueCache
  // This function is thread safe
  template <typename Queue>
  QueueCache<Queue>& getQueueCache() {
    // the public interface is thread safe
    static QueueCache<Queue> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_interface_QueueCache_h
