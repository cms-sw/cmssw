#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Digis_test.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

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
      SiPixelDigisSoACollection digis_d(1000, queue);
      testDigisSoA::runKernels(digis_d.view(), queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      SiPixelDigisHost digis_h(digis_d.view().metadata().size(), queue);

      std::cout << digis_h.view().metadata().size() << std::endl;
      alpaka::memcpy(queue, digis_h.buffer(), digis_d.const_buffer());
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}
