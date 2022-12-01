#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

namespace testTrackingRecHit2D {

  template <typename TrackerTraits>
  __global__ void fill(TrackingRecHit2DSOAViewT<TrackerTraits>* phits) {
    assert(phits);
    auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  template <typename TrackerTraits>
  __global__ void verify(TrackingRecHit2DSOAViewT<TrackerTraits> const* phits) {
    assert(phits);
    auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }
}  // namespace testTrackingRecHit2D
