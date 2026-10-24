#ifndef HeterogeneousCore_AlpakaInterface_interface_FixedQueueRegistry_h
#define HeterogeneousCore_AlpakaInterface_interface_FixedQueueRegistry_h

#include <alpaka/alpaka.hpp>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace cms::alpakatools {

  template <typename Queue>
  class FixedQueueRegistry {
  public:
    using Device = decltype(alpaka::getDev(std::declval<Queue>()));
    using QueueHandle = decltype(alpaka::getNativeHandle(std::declval<Queue>()));
    using Platform = alpaka::Platform<Device>;

    struct Key {
      QueueHandle queue;

      bool operator==(Key const& other) const { return queue == other.queue; }
    };

    struct KeyHash {
      std::size_t operator()(Key const& key) const noexcept { return std::hash<QueueHandle>{}(key.queue); }
    };

    using Registry = std::unordered_map<Key, Queue, KeyHash>;

    FixedQueueRegistry() : registry_(devices<Platform>().size()) {}

    void registerQueue(Queue const& queue) {
      auto const deviceId = alpaka::getNativeHandle(alpaka::getDev(queue));
      auto const nativeQueue = alpaka::getNativeHandle(queue);

      std::scoped_lock lock(mutex_);
      registry_.at(deviceId).try_emplace(Key{nativeQueue}, queue);
    }

    void unregisterQueue(Queue const& queue) {
      auto const deviceId = alpaka::getNativeHandle(alpaka::getDev(queue));
      auto const nativeQueue = alpaka::getNativeHandle(queue);

      std::scoped_lock lock(mutex_);
      registry_.at(deviceId).erase(Key{nativeQueue});
    }

    std::optional<Queue> findQueue(int deviceId, QueueHandle nativeQueue) const {
      std::scoped_lock lock(mutex_);

      if (deviceId < 0 || static_cast<std::size_t>(deviceId) >= registry_.size()) {
        return std::nullopt;
      }

      auto const& queues = registry_.at(deviceId);
      auto const it = queues.find(Key{nativeQueue});

      if (it != queues.end()) {
        return it->second;
      }

      return std::nullopt;
    }

    void clear() {
      std::scoped_lock lock(mutex_);
      registry_.assign(devices<Platform>().size(), {});
    }

  private:
    std::vector<Registry> registry_;
    mutable std::mutex mutex_;
  };

  template <typename Queue>
  FixedQueueRegistry<Queue>& getFixedQueueRegistry() {
    CMS_THREAD_SAFE static FixedQueueRegistry<Queue> registry;
    return registry;
  }
}  // namespace cms::alpakatools
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED or ALPAKA_ACC_GPU_HIP_ENABLED
#endif  // HeterogeneousCore_AlpakaInterface_interface_FixedQueueRegistry_h
