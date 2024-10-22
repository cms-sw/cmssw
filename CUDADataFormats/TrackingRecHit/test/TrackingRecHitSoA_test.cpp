#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackingRecHitSoA {

  template <typename TrackerTraits>
  void runKernels(TrackingRecHitSoADevice<TrackerTraits>& hits, cudaStream_t stream);

}

int main() {
  using ParamsOnGPU = TrackingRecHitSoADevice<pixelTopology::Phase1>::ParamsOnGPU;
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

  // inner scope to deallocate memory before destroying the stream
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[1856];

    for (size_t i = 0; i < 1856; i++) {
      moduleStart[i] = i * 2;
    }
    ParamsOnGPU* cpeParams_d;
    cudaCheck(cudaMalloc(&cpeParams_d, sizeof(ParamsOnGPU)));
    TrackingRecHitSoADevice<pixelTopology::Phase1> tkhit(nHits, offset, cpeParams_d, &moduleStart[0], stream);

    testTrackingRecHitSoA::runKernels<pixelTopology::Phase1>(tkhit, stream);
    printf("tkhit hits %d \n", tkhit.nHits());
    auto test = tkhit.localCoordToHostAsync(stream);
    printf("test[9] %.2f\n", test[9]);

    auto ret = tkhit.hitsModuleStartToHostAsync(stream);
    printf("mods[9] %d\n", ret[9]);
    cudaCheck(cudaFree(cpeParams_d));
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
