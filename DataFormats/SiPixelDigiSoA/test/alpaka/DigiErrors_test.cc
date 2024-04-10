#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

#include "DigiErrors_test.h"

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
      SiPixelDigiErrorsSoACollection digiErrors_d(1000, queue);
      testDigisSoA::runKernels(digiErrors_d.view(), queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      SiPixelDigiErrorsHost digiErrors_h(digiErrors_d.view().metadata().size(), queue);
      alpaka::memcpy(queue, digiErrors_h.buffer(), digiErrors_d.const_buffer());
      std::cout << "digiErrors_h.view().metadata().size(): " << digiErrors_h.view().metadata().size() << std::endl;
      std::cout << "digiErrors_h.view()[100].pixelErrors().rawId: " << digiErrors_h.view()[100].pixelErrors().rawId
                << std::endl;
      std::cout << "digiErrors_h.view()[100].pixelErrors().word: " << digiErrors_h.view()[100].pixelErrors().word
                << std::endl;
      std::cout << "digiErrors_h.view()[100].pixelErrors().errorType: "
                << digiErrors_h.view()[100].pixelErrors().errorType << std::endl;
      std::cout << "digiErrors_h.view()[100].pixelErrors().fedId: " << digiErrors_h.view()[100].pixelErrors().fedId
                << std::endl;
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}
