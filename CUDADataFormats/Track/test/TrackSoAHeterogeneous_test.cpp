/**
   Simple test for the pixelTrack::TrackSoA data structure
   which inherits from PortableDeviceCollection.

   Creates an instance of the class (automatically allocates
   memory on device), passes the view of the SoA data to
   the CUDA kernels which:
   - Fill the SoA with data.
   - Verify that the data written is correct.

   Then, the SoA data are copied back to Host, where
   a temporary host-side view (tmp_view) is created using
   the same Layout to access the data on host and print it.
 */

#include <cstdint>
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackSoA {

  template <typename TrackerTraits>
  void runKernels(TrackSoAView<TrackerTraits> &tracks_view, cudaStream_t stream);
}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on device. PortableDeviceCollection allocates
    // SoA on device automatically.
    TrackSoAHeterogeneousDevice<pixelTopology::Phase1> tracks_d(stream);
    testTrackSoA::runKernels<pixelTopology::Phase1>(tracks_d.view(), stream);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    TrackSoAHeterogeneousHost<pixelTopology::Phase1> tracks_h(stream);

    cudaCheck(cudaMemcpyAsync(
        tracks_h.buffer().get(), tracks_d.const_buffer().get(), tracks_d.bufferSize(), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    // Print results
    std::cout << "pt"
              << "\t"
              << "eta"
              << "\t"
              << "chi2"
              << "\t"
              << "quality"
              << "\t"
              << "nLayers"
              << "\t"
              << "hitIndices off" << std::endl;

    for (int i = 0; i < 10; ++i) {
      std::cout << tracks_h.view()[i].pt() << "\t" << tracks_h.view()[i].eta() << "\t" << tracks_h.view()[i].chi2()
                << "\t" << (int)tracks_h.view()[i].quality() << "\t" << (int)tracks_h.view()[i].nLayers() << "\t"
                << tracks_h.view().hitIndices().off[i] << std::endl;
    }
  }
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
