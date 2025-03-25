#include <alpaka/alpaka.hpp>

#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersHost.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "FWCore/Utilities/interface/stringize.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

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
      // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClustersDevice clusters_d(testSiStripClusterSoA::kMaxSeedStrips,
                                                                              queue);

      // Fill and verify on device
      testSiStripClusterSoA::runKernels(clusters_d.view(), queue);

      // Instantate tracks on host. This is where the data will be copied to from device.
      // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClustersHost clusters_h(testSiStripClusterSoA::kMaxSeedStrips,
                                                                            queue);

      // Copy on host
      alpaka::memcpy(queue, clusters_h.buffer(), clusters_d.const_buffer());
      alpaka::wait(queue);

      // Checks that on the host the objects verify same conditions of ALPAKA_ACCELERATOR_NAMESPACE::testSiStripClusterSoA::TestVerifyKernel
      for (uint32_t j = 0; j < testSiStripClusterSoA::kMaxSeedStrips; ++j) {
        assert(clusters_h->clusterIndex(j) == j);
        assert(clusters_h->clusterSize(j) == j * 2);
        for (int k = 0; k < 768; ++k) {
          assert(clusters_h->clusterADCs(j)[k] == (uint8_t)((j + k) % 255));
        }
        assert(clusters_h->clusterDetId(j) == j + 12);
        assert(clusters_h->firstStrip(j) == j % 65536);
        assert(clusters_h->trueCluster(j) == (j % 2 == 0));
        assert(clusters_h->barycenter(j) == j * 1.0f);
        assert(clusters_h->charge(j) == j * -1.0f);
      }
      // set this only once in the whole kernel grid
      assert(clusters_h->nClusters() == testSiStripClusterSoA::kMaxSeedStrips);
      assert(clusters_h->maxClusterSize() == ALPAKA_ACCELERATOR_NAMESPACE::sistrip::maxStripsPerCluster);
    }  // namespace
  }
  return EXIT_SUCCESS;
}
