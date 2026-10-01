// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef HeterogeneousCore_AlpakaInterface_interface_CachingAllocatorMonitor_h
#define HeterogeneousCore_AlpakaInterface_interface_CachingAllocatorMonitor_h

#include <atomic>
#include <cstddef>

namespace cms::alpakatools {

  // Optional, process-wide observer of CachingAllocator transactions.
  //
  // Free of alpaka and profiler dependencies. With no monitor registered the
  // allocator pays one atomic load and a not-taken branch per transaction.
  // A registered monitor must outlive all allocator use; its callbacks run on
  // the allocating/freeing thread with the allocator mutex held, so they must be
  // thread-safe and cheap.
  class CachingAllocatorMonitor {
  public:
    // The allocator's memory space: the host (pinned) allocator, or a device one
    // with the backend's native device index (e.g. the CUDA ordinal).
    struct Device {
      int index;
      bool host;
    };

    virtual ~CachingAllocatorMonitor() = default;

    // A block was handed out: |bytes| is the bin-rounded size, |requested| the
    // user size, |cacheHit| tells whether a cached block was reused, |queue|
    // identifies the associated queue.
    virtual void onAllocate(Device device,
                            void const* ptr,
                            std::size_t bytes,
                            std::size_t requested,
                            bool cacheHit,
                            unsigned long long queue) noexcept {}

    // A block was returned. Device work on |queue| may still use it; the
    // allocator re-hands it only after an event recorded on |queue| completes.
    virtual void onFree(Device device, void const* ptr, std::size_t bytes, unsigned long long queue) noexcept {}

    // The allocator's byte totals after a transaction.
    virtual void onUsage(Device device, std::size_t live, std::size_t cached, std::size_t requested) noexcept {}
  };

  inline std::atomic<CachingAllocatorMonitor*>& cachingAllocatorMonitorRef() noexcept {
    static std::atomic<CachingAllocatorMonitor*> instance{nullptr};
    return instance;
  }

  inline void setCachingAllocatorMonitor(CachingAllocatorMonitor* monitor) noexcept {
    cachingAllocatorMonitorRef().store(monitor, std::memory_order_release);
  }

  inline CachingAllocatorMonitor* cachingAllocatorMonitor() noexcept {
    return cachingAllocatorMonitorRef().load(std::memory_order_acquire);
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_CachingAllocatorMonitor_h
