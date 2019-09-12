#include "HeterogeneousCore/CUDAUtilities/interface/CUDAEventCache.h"

namespace cudautils {
  // CUDAEventCache should be constructed by the first call to
  // getCUDAEventCache() only if we have CUDA devices present
  CUDAEventCache::CUDAEventCache() : cache_(cuda::device::count()) {}

  std::shared_ptr<cuda::event_t> CUDAEventCache::getCUDAEvent() {
    return cache_[cuda::device::current::get().id()].makeOrGet([]() {
      auto current_device = cuda::device::current::get();
      // We should not return a recorded, but not-yet-occurred event
      return std::make_unique<cuda::event_t>(
          current_device
              .create_event(  // default; we should try to avoid explicit synchronization, so maybe the value doesn't matter much?
                  cuda::event::sync_by_busy_waiting,
                  // it should be a bit faster to ignore timings
                  cuda::event::dont_record_timings));
    });
  }

  void CUDAEventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // CUDAEventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(cuda::device::count());
  }

  CUDAEventCache& getCUDAEventCache() {
    static CUDAEventCache cache;
    return cache;
  }
}  // namespace cudautils
