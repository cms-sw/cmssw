#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "FWCore/Utilities/interface/stringize.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"

#include "TestSiStripClusterizerConditionsDevice.h"

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
      // Assuming reasonable sizes for these structures
      constexpr const unsigned int DetToFeds_size = 34813;
      constexpr const unsigned int Data_fedch_size = 42240;
      constexpr const unsigned int Data_strip_size = 10813440;
      constexpr const unsigned int Data_apv_size = 84480;
      SiStripClusterizerConditionsDevice conditions_d(
          {{DetToFeds_size, Data_fedch_size, Data_strip_size, Data_apv_size}},
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)

      testConditionsSoA::runKernels(conditions_d.view(),
                                    conditions_d.view<SiStripClusterizerConditionsData_fedchSoA>(),
                                    conditions_d.view<SiStripClusterizerConditionsData_stripSoA>(),
                                    conditions_d.view<SiStripClusterizerConditionsData_apvSoA>(),
                                    queue);

      SiStripClusterizerConditionsHost conditions_h(conditions_d.sizes(), queue);
      alpaka::memcpy(queue, conditions_h.buffer(), conditions_d.const_buffer());
      alpaka::wait(queue);

      for (uint32_t j = 0; j < (uint32_t)conditions_d.sizes()[0]; ++j) {
        assert(conditions_h->detid_(j) == j * 2);
        assert(conditions_h->ipair_(j) == (uint16_t)((j) % 65536));
        assert(conditions_h->fedid_(j) == (uint16_t)((j + 1) % 65536));
        assert(conditions_h->fedch_(j) == (uint8_t)(j % 256));
      }

      for (uint32_t j = 0; j < (uint32_t)conditions_d.sizes()[1]; ++j) {
        assert(conditions_h.view<SiStripClusterizerConditionsData_fedchSoA>().detID_(j) == (uint32_t)(j));
        assert(conditions_h.view<SiStripClusterizerConditionsData_fedchSoA>().iPair_(j) == (uint16_t)(j % 65536));
        assert(conditions_h.view<SiStripClusterizerConditionsData_fedchSoA>().invthick_(j) == (float)(j * 1.0));
      }

      for (uint32_t j = 0; j < (uint32_t)conditions_d.sizes()[2]; ++j) {
        assert(conditions_h.view<SiStripClusterizerConditionsData_stripSoA>().noise_(j) == (uint16_t)(j % 65536));
      }

      for (uint32_t j = 0; j < (uint32_t)conditions_d.sizes()[3]; ++j) {
        assert(conditions_h.view<SiStripClusterizerConditionsData_apvSoA>().gain_(j) == (float)(j * -1.0f));
      }
    }
  }

  return EXIT_SUCCESS;
}
