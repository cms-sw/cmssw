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

#include <alpaka/alpaka.hpp>
#include <unistd.h>
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

using namespace std;
using namespace reco;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelTrack;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testTrackSoA {

    template <typename TrackerTraits>
    void runKernels(TrackSoAView<TrackerTraits> tracks_view, Queue& queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on device. PortableDeviceCollection allocates
    // SoA on device automatically.
    TracksSoACollection<pixelTopology::Phase1> tracks_d(queue);
    testTrackSoA::runKernels<pixelTopology::Phase1>(tracks_d.view(), queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    TracksHost<pixelTopology::Phase1> tracks_h(queue);

    std::cout << tracks_h.view().metadata().size() << std::endl;
    alpaka::memcpy(queue, tracks_h.buffer(), tracks_d.const_buffer());
    alpaka::wait(queue);

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

  return 0;
}
