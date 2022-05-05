#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

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
    TrackingRecHit2DGPU tkhit(nHits, false, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels(tkhit.view());

    TrackingRecHit2DGPU tkhitPhase2(nHits, true, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels(tkhitPhase2.view());

    TrackingRecHit2DHost tkhitH(nHits, false, 0, nullptr, nullptr, stream, &tkhit);

    memoryPool::cuda::dumpStat();

    cudaStreamSynchronize(stream);
    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));
    assert(tkhitH.view()->nMaxModules() == phase1PixelTopology::numberOfModules);

    TrackingRecHit2DHost tkhitHPhase2(nHits, true, 0, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitHPhase2.view());
    assert(tkhitHPhase2.view()->nHits() == unsigned(nHits));
    assert(tkhitHPhase2.view()->nMaxModules() == phase2PixelTopology::numberOfModules);

    memoryPool::cuda::dumpStat();
  }

   cudaCheck(cudaStreamSynchronize(stream));
   memoryPool::cuda::dumpStat();

   std::cout <<    "on CPU" << std::endl;
   ((SimplePoolAllocatorImpl<PosixAlloc>*)memoryPool::cuda::getPool(memoryPool::onCPU))->dumpStat();

   cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
