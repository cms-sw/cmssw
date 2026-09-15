// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef HeterogeneousCore_AlpakaInterface_interface_CachingAllocatorMonitor_h
#define HeterogeneousCore_AlpakaInterface_interface_CachingAllocatorMonitor_h

#include <atomic>
#include <cstddef>

namespace cms::alpakatools {

  // Optional, process-wide hook to observe caching-allocator transactions.
  //
  // Kept free of any alpaka or perfetto dependency so that the allocator (which
  // calls it on the hot path) stays cheap -- when no monitor is registered the
  // cost is a single load of an atomic pointer (a read-shared global, hence a
  // plain MOV on x86) plus a predicted-not-taken branch -- and so that an external
  // profiler can implement it without pulling alpaka headers. A registered monitor must
  // outlive all allocator use and its callbacks must be thread-safe: they are
  // invoked from whatever TBB worker (or async/GPU callback thread) runs the
  // owning module, while the allocator's internal mutex is held.
  class CachingAllocatorMonitor {
  public:
    virtual ~CachingAllocatorMonitor() = default;

    // A block was handed out. |bytes| is the bin-rounded size, |requested| the
    // user size, |cacheHit| is true when a cached block was reused (no new
    // device allocation), |queue| identifies the associated backend queue/stream
    // and |device| is the native device handle (e.g. CUDA ordinal).
    virtual void onAllocate(int device,
                            const void* ptr,
                            std::size_t bytes,
                            std::size_t requested,
                            bool cacheHit,
                            unsigned long long queue) noexcept {}

    // A block was returned to the allocator. Note the memory may still be in use
    // by asynchronous device work; the allocator records an event on |queue| and
    // only re-hands the block once that event completes (possibly to another
    // thread/stream) -- this is the "asynchronous transaction".
    virtual void onFree(int device, const void* ptr, std::size_t bytes, unsigned long long queue) noexcept {}

    // Running byte totals for |device| right after a transaction.
    virtual void onUsage(int device, std::size_t live, std::size_t cached, std::size_t requested) noexcept {}
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
