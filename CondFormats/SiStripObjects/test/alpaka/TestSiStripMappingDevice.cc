#include <cstdlib>

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiStripObjects/interface/SiStripMappingSoA.h"
#include "CondFormats/SiStripObjects/interface/SiStripMappingHost.h"
#include "CondFormats/SiStripObjects/interface/alpaka/SiStripMappingDevice.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestSiStripMappingDevice.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

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
      SiStripMappingDevice conditions_d(
          100, queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      testMappingSoA::runKernels(conditions_d.view(), queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      SiStripMappingHost conditions_h(
          conditions_d.view().metadata().size(),
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      std::cout << "conditions_h.view().metadata().size() = " << conditions_h.view().metadata().size() << std::endl;
      alpaka::memcpy(queue, conditions_h.buffer(), conditions_d.const_buffer());
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}