#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "TrackingRecHit2DCUDAImpl_t.h"

namespace testTrackingRecHit2D {

  template <typename TrackerTraits>
  void runKernels(TrackingRecHit2DSOAViewT<TrackerTraits>* hits) {
    assert(hits);
    fill<TrackerTraits><<<1, 1024>>>(hits);
    verify<TrackerTraits><<<1, 1024>>>(hits);
  }

  template void runKernels<pixelTopology::Phase1>(TrackingRecHit2DSOAViewT<pixelTopology::Phase1>* hits);
  template void runKernels<pixelTopology::Phase2>(TrackingRecHit2DSOAViewT<pixelTopology::Phase2>* hits);
}  // namespace testTrackingRecHit2D
