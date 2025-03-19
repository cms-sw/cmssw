#include <cstdlib>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersHost.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestSiStripClustersDevice.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
// using namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip;

int main() {
  // Get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(
        ALPAKA_ACCELERATOR_NAMESPACE) " backend, the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device
  for (const auto& device : devices) {
    Queue queue(device);

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate tracks on device. PortableDeviceCollection allocates
      // SoA on device automatically.
      ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClustersDevice clusters_d(
          testSiStripClusterSoA::kMaxSeedStrips,
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      testSiStripClusterSoA::runKernels(clusters_d.view(), queue);

      // Instantate tracks on host. This is where the data will be copied to from device.
      ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClustersHost clusters_h(
          testSiStripClusterSoA::kMaxSeedStrips,
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      alpaka::memcpy(queue, clusters_h.buffer(), clusters_d.const_buffer());
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}
