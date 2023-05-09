#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

namespace testTrackingRecHitSoA {

  template <typename TrackerTraits>
  __global__ void fill(TrackingRecHitSoAView<TrackerTraits> soa) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    if (i == 0 and j == 0) {
      soa.offsetBPIX2() = 22;
      soa[10].xLocal() = 1.11;
    }

    soa[i].iphi() = i % 10;
    soa.hitsLayerStart()[j] = j;
    __syncthreads();
  }

  template <typename TrackerTraits>
  __global__ void show(TrackingRecHitSoAView<TrackerTraits> soa) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    if (i == 0 and j == 0) {
      printf("nbins = %d \n", soa.phiBinner().nbins());
      printf("offsetBPIX %d ->%d \n", i, soa.offsetBPIX2());
      printf("nHits %d ->%d \n", i, soa.nHits());
      printf("hitsModuleStart %d ->%d \n", i, soa.hitsModuleStart().at(28));
    }

    if (i < 10)  // can be increased to soa.nHits() for debugging
      printf("iPhi %d ->%d \n", i, soa[i].iphi());

    if (j * blockDim.x + i < 10)  // can be increased to soa.phiBinner().nbins() for debugging
      printf(">bin size %d ->%d \n", j * blockDim.x + i, soa.phiBinner().size(j * blockDim.x + i));
    __syncthreads();
  }

  template <typename TrackerTraits>
  void runKernels(TrackingRecHitSoADevice<TrackerTraits>& hits, cudaStream_t stream) {
    printf("> RUN!\n");
    fill<TrackerTraits><<<10, 100, 0, stream>>>(hits.view());

    cudaCheck(cudaDeviceSynchronize());
    cms::cuda::fillManyFromVector(&(hits.view().phiBinner()),
                                  10,
                                  hits.view().iphi(),
                                  hits.view().hitsLayerStart().data(),
                                  2000,
                                  256,
                                  hits.view().phiBinnerStorage(),
                                  stream);
    cudaCheck(cudaDeviceSynchronize());
    show<TrackerTraits><<<10, 1000, 0, stream>>>(hits.view());
    cudaCheck(cudaDeviceSynchronize());
  }

  template void runKernels<pixelTopology::Phase1>(TrackingRecHitSoADevice<pixelTopology::Phase1>& hits,
                                                  cudaStream_t stream);
  template void runKernels<pixelTopology::Phase2>(TrackingRecHitSoADevice<pixelTopology::Phase2>& hits,
                                                  cudaStream_t stream);

}  // namespace testTrackingRecHitSoA
