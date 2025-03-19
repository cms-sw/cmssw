#include <cstdlib>

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/alpaka/SiStripClusterizerConditionsDevice.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestSiStripClusterizerConditionsDevice.h"

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
      int const DetToFedsSoA_size = 10;
      int const Data_fedch_SoA_size = 50;
      int const Data_strip_SoA_size = 100;
      int const Data_apv_SoA_size = 200;
      SiStripClusterizerConditionsDevice conditions_d(
          {{DetToFedsSoA_size, Data_fedch_SoA_size, Data_strip_SoA_size, Data_apv_SoA_size}},
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      testConditionsSoA::runKernels(conditions_d.view(),
                                    conditions_d.view<SiStripClusterizerConditionsData_fedchSoA>(),
                                    conditions_d.view<SiStripClusterizerConditionsData_stripSoA>(),
                                    conditions_d.view<SiStripClusterizerConditionsData_apvSoA>(),
                                    queue);

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      int const DetToFedsSoA_sizeFromMeta = conditions_d.view().metadata().size();
      int const Data_fedch_SoA_sizeFromMeta =
          conditions_d.view<SiStripClusterizerConditionsData_fedchSoA>().metadata().size();
      int const Data_strip_SoA_sizeFromMeta =
          conditions_d.view<SiStripClusterizerConditionsData_stripSoA>().metadata().size();
      int const Data_apv_SoA_sizeFromMeta =
          conditions_d.view<SiStripClusterizerConditionsData_apvSoA>().metadata().size();

      SiStripClusterizerConditionsHost conditions_h(
          {{DetToFedsSoA_sizeFromMeta,
            Data_fedch_SoA_sizeFromMeta,
            Data_strip_SoA_sizeFromMeta,
            Data_apv_SoA_sizeFromMeta}},
          queue);  // (the namespace specification is to avoid confusion with the non-alpaka sistrip namespace)
      std::cout << "conditions_h.view().metadata().size() = " << conditions_h.view().metadata().size() << std::endl;
      alpaka::memcpy(queue, conditions_h.buffer(), conditions_d.const_buffer());
      alpaka::wait(queue);
    }
  }

  return EXIT_SUCCESS;
}