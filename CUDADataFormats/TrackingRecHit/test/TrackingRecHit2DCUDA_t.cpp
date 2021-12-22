#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView* hits);

}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto nHits = 200;
  auto nModules = 2000;
  // inner scope to deallocate memory before destroying the stream
  {
    TrackingRecHit2DGPU tkhit(nHits, nModules, 0, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.view());

    TrackingRecHit2DHost tkhitH(nHits, nModules, 0, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));
    assert(tkhitH.view()->nMaxModules() == unsigned(nModules));
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
