#include <alpaka/alpaka.hpp>
#include "FWCore/Utilities/interface/stringize.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripMappingDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingSoA.h"

#include "TestSiStripMappingDevice.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip;

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
      // A reasonable size for the collection is picked as O(strips), which should correspond to the the MappingDevice from conditions
      constexpr const unsigned int nStrips = 200000;
      SiStripMappingDevice conditions_d(nStrips, queue);

      testMappingSoA::runKernels(conditions_d.view(), queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      SiStripMappingHost conditions_h(
          conditions_d.view().metadata().size(),
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      alpaka::memcpy(queue, conditions_h.buffer(), conditions_d.const_buffer());
      alpaka::wait(queue);

      // Check on host that all verified on the device also passes assertions here
      for (uint32_t j = 0; j < nStrips; ++j) {
        // assert(conditions_h->input(j) - arr == j % 10); this is supposed to be a pointer on the device memory space
        assert(conditions_h->inoff(j) == (size_t)j);
        assert(conditions_h->offset(j) == (size_t)j);
        assert(conditions_h->length(j) == (uint16_t)(j % 65536));
        assert(conditions_h->fedID(j) == (uint16_t)(j % 65536));
        assert(conditions_h->fedCh(j) == (uint8_t)(j % 256));
        assert(conditions_h->detID(j) == 3 * j);
      }
    }
  }

  return EXIT_SUCCESS;
}
