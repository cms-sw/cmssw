/* Simple test for the copyFromDense and copyToDense utilities from DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h .
 *
 * Creates an instance of TracksSoACollection<pixelTopology::Phase1> (automatically allocates memory on device),
 * passes the view of the SoA data to the kernel that:
 *   - fill the SoA with covariance data;
 *   - copy the covariance data to the dense representation, and back to the matrix representation;
 *   - verify that the data is copied back and forth correctly.
 */

#include <cstdlib>
#include <iostream>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

#include "TrajectoryStateSoA_t.h"

// Each test binary is built for a single Alpaka backend.
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;
int main() {
  // Get the list of devices on the current platform.
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device.
  for (const auto& device : devices) {
    Queue queue(device);

    // Inner scope to deallocate memory before destroying the stream.
    {
      TracksSoACollection tracks_d({{1000,5000}},queue);

      test::testTrackSoA<pixelTopology::Phase1>(queue, tracks_d.view());

      // Wait for the tests to complete.
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}
