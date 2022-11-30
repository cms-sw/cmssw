#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackingRecHit2D {

  template <typename TrackerTraits>
  void runKernels(TrackingRecHit2DSOAViewT<TrackerTraits>* hits);
}  // namespace testTrackingRecHit2D

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto nHits = 200;
  // inner scope to deallocate memory before destroying the stream
  {
    TrackingRecHit2DGPUT<pixelTopology::Phase1> tkhit(nHits, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels<pixelTopology::Phase1>(tkhit.view());

    TrackingRecHit2DGPUT<pixelTopology::Phase2> tkhitPhase2(nHits, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels<pixelTopology::Phase2>(tkhitPhase2.view());

    TrackingRecHit2DHostT<pixelTopology::Phase1> tkhitH(nHits, 0, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));

    TrackingRecHit2DHostT<pixelTopology::Phase2> tkhitHPhase2(nHits, 0, nullptr, nullptr, stream, &tkhitPhase2);
    cudaStreamSynchronize(stream);
    assert(tkhitHPhase2.view());
    assert(tkhitHPhase2.view()->nHits() == unsigned(nHits));
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
