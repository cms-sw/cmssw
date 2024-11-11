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

#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TrackSoAHeterogeneous_test.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

int main() {
  // Get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device
  for (const auto& device : devices) {
    Queue queue(device);

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate tracks on device. PortableDeviceCollection allocates
      // SoA on device automatically.
      constexpr auto nTracks = 1000;
      constexpr auto nHits = nTracks *  5;

      TracksSoACollection tracks_d({{nTracks,nHits}},queue);
      testTrackSoA::runKernels<pixelTopology::Phase1>(tracks_d.view(), queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      ::reco::TracksHost tracks_h({{nTracks,nHits}},queue);

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
                  << tracks_h.view()[i].hitOffsets() << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
