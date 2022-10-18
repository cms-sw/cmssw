#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "TrackingRecHit2DCUDAImpl_t.h"

namespace testTrackingRecHit2D {

  // template <typename TrackerTraits>
  void runKernels(TrackingRecHit2DSOAViewT<pixelTopology::Phase1>* hits) {
    assert(hits);
    fill<pixelTopology::Phase1><<<1, 1024>>>(hits);
    verify<pixelTopology::Phase1><<<1, 1024>>>(hits);
  }

  void runKernelsPhase2(TrackingRecHit2DSOAViewT<pixelTopology::Phase2>* hits) {
    assert(hits);
    fill<pixelTopology::Phase2><<<1, 1024>>>(hits);
    verify<pixelTopology::Phase2><<<1, 1024>>>(hits);
  }

}  // namespace testTrackingRecHit2D
