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
  // inner scope to deallocate memory before destroying the stream
  {
    TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.view());

    TrackingRecHit2DHost tkhitH(nHits, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
