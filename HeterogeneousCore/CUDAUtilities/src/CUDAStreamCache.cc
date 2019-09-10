#include "HeterogeneousCore/CUDAUtilities/interface/CUDAStreamCache.h"

namespace cudautils {
  // CUDAStreamCache should be constructed by the first call to
  // getCUDAStreamCache() only if we have CUDA devices present
  CUDAStreamCache::CUDAStreamCache()
    : cache_(cuda::device::count()) {}


  std::shared_ptr<cuda::stream_t<>> CUDAStreamCache::getCUDAStream() {
    return cache_[cuda::device::current::get().id()].makeOrGet([](){
        auto current_device = cuda::device::current::get();
        return std::make_unique<cuda::stream_t<>>(current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream));
      });
  }

  void CUDAStreamCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // CUDAStreamCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(cuda::device::count());
  }

  CUDAStreamCache& getCUDAStreamCache() {
    static CUDAStreamCache cache;
    return cache;
  }
}
