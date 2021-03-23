#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

namespace testTrackingRecHit2D {

  __global__ void fill(TrackingRecHit2DSOAView* phits) {
    assert(phits);
    auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  __global__ void verify(TrackingRecHit2DSOAView const* phits) {
    assert(phits);
    auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  void runKernels(TrackingRecHit2DSOAView* hits) {
    assert(hits);
    fill<<<1, 1024>>>(hits);
    verify<<<1, 1024>>>(hits);
  }

}  // namespace testTrackingRecHit2D
