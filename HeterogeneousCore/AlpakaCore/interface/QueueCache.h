#ifndef HeterogeneousCore_AlpakaCore_interface_QueueCache_h
#define HeterogeneousCore_AlpakaCore_interface_QueueCache_h

#include <memory>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

namespace cms::alpakatools {

  template <typename Queue>
  class QueueCache {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    friend class alpaka_cuda_async::AlpakaService;
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    friend class alpaka_rocm_async::AlpakaService;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    friend class alpaka_serial_sync::AlpakaService;
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    friend class alpaka_tbb_async::AlpakaService;
#endif

    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Platform<Device>;

  public:
    // QueueCache should be constructed by the first call to
    // getQueueCache() only if we have any devices present
    QueueCache() : cache_() {
      // FIXME do not hardcode
      init(32);
    }

    // Gets a (cached) queue for the current device. The queue
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<Queue> get(Device const& dev) { return cache_[alpaka::getNativeHandle(dev)]; }

    std::shared_ptr<Queue> get(edm::StreamID sid) { return cache_[sid.value()]; }

  private:
    // not thread safe, intended to be called only from AlpakaService
    void clear() { cache_.clear(); }

    void init(int streams) {
      int size = std::max<int>(streams, devices<Platform>().size());
      cache_.clear();
      cache_.reserve(size);
      for (int i = 0; i < size; ++i) {
        // Copied from HeterogeneousCore/AlpakaCore/src/alpaka/chooseDevice.cc
        // to get a consistent behaviour.
        int id = i % devices<Platform>().size();
        const Device& device = devices<Platform>()[id];
        cache_.emplace_back(std::make_shared<Queue>(device));
      }
    }

    std::vector<std::shared_ptr<Queue>> cache_;
  };

  // Gets the global instance of a QueueCache
  // This function is thread safe
  template <typename Queue>
  QueueCache<Queue>& getQueueCache() {
    // the public interface is thread safe
    CMS_THREAD_SAFE static QueueCache<Queue> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_interface_QueueCache_h
