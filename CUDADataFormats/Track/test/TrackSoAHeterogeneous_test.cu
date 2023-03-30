#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace testTrackSoA {

  // Kernel which fills the TrackSoAView with data
  // to test writing to it
  template <typename TrackerTraits>
  __global__ void fill(TrackSoAView<TrackerTraits> tracks_view) {
    int i = threadIdx.x;
    if (i == 0) {
      tracks_view.nTracks() = 420;
    }

    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      tracks_view[j].pt() = (float)j;
      tracks_view[j].eta() = (float)j;
      tracks_view[j].chi2() = (float)j;
      tracks_view[j].quality() = (pixelTrack::Quality)(j % 256);
      tracks_view[j].nLayers() = j % 128;
      tracks_view.hitIndices().off[j] = j;
    }
  }

  // Kernel which reads from the TrackSoAView to verify
  // that it was written correctly from the fill kernel
  template <typename TrackerTraits>
  __global__ void verify(TrackSoAConstView<TrackerTraits> tracks_view) {
    int i = threadIdx.x;

    if (i == 0) {
      printf("SoA size: % d, block dims: % d\n", tracks_view.metadata().size(), blockDim.x);
      assert(tracks_view.nTracks() == 420);
    }
    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      assert(abs(tracks_view[j].pt() - (float)j) < .0001);
      assert(abs(tracks_view[j].eta() - (float)j) < .0001);
      assert(abs(tracks_view[j].chi2() - (float)j) < .0001);
      assert(tracks_view[j].quality() == (pixelTrack::Quality)(j % 256));
      assert(tracks_view[j].nLayers() == j % 128);
      assert(tracks_view.hitIndices().off[j] == j);
    }
  }

  // Host function which invokes the two kernels above
  template <typename TrackerTraits>
  void runKernels(TrackSoAView<TrackerTraits>& tracks_view, cudaStream_t stream) {
    fill<TrackerTraits><<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    verify<TrackerTraits><<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
  }

  template void runKernels<pixelTopology::Phase1>(TrackSoAView<pixelTopology::Phase1>& tracks_view,
                                                  cudaStream_t stream);
  template void runKernels<pixelTopology::Phase2>(TrackSoAView<pixelTopology::Phase2>& tracks_view,
                                                  cudaStream_t stream);

}  // namespace testTrackSoA
