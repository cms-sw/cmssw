#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testClusterSoA {

    void runKernels(SiPixelClustersSoAView clust_view, Queue& queue);
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
    SiPixelClustersSoACollection clusters_d(100, queue);
    testClusterSoA::runKernels(clusters_d.view(), queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    SiPixelClustersHost clusters_h(clusters_d.view().metadata().size(), queue);

    std::cout << clusters_h.view().metadata().size() << std::endl;
    alpaka::memcpy(queue, clusters_h.buffer(), clusters_d.const_buffer());
    alpaka::wait(queue);
  }

  return 0;
}
