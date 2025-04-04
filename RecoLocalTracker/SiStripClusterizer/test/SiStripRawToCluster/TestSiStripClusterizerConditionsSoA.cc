// A minimal test to ensure that
//   - sistrip::SiStripClusterizerConditionsSoA, sistrip::SiStripClusterizerConditionsHost can be compiled
//   - sistrip::SiStripClusterizerConditionsHost can be allocated, modified and erased (on host)
//   - view-based element access works

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsHost.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"

using namespace sistrip;

int main() {
  // Assuming reasonable sizes for these structures
  constexpr const unsigned int DetToFeds_size = 34813;
  constexpr const unsigned int Data_fedch_size = 42240;
  constexpr const unsigned int Data_strip_size = 10813440;
  constexpr const unsigned int Data_apv_size = 84480;
  SiStripClusterizerConditionsDetToFedsHost sStripCond_DetToFeds(DetToFeds_size, cms::alpakatools::host());
  SiStripClusterizerConditionsDataHost sStripCond_Data({{Data_fedch_size, Data_strip_size, Data_apv_size}},
                                                       cms::alpakatools::host());

  auto Data_fedch_View = sStripCond_Data.view<SiStripClusterizerConditionsData_fedchSoA>();
  auto Data_strip_View = sStripCond_Data.view<SiStripClusterizerConditionsData_stripSoA>();
  auto Data_apv_View = sStripCond_Data.view<SiStripClusterizerConditionsData_apvSoA>();

  for (uint32_t j = 0; j < DetToFeds_size; j++) {
    sStripCond_DetToFeds->detid_(j) = j * 2;
    sStripCond_DetToFeds->ipair_(j) = (uint16_t)((j) % 65536);
    sStripCond_DetToFeds->fedid_(j) = (uint16_t)((j + 1) % 65536);
    sStripCond_DetToFeds->fedch_(j) = (uint8_t)(j % 256);
  }

  for (uint32_t j = 0; j < Data_fedch_size; j++) {
    Data_fedch_View.detID_(j) = (uint32_t)(j);
    Data_fedch_View.iPair_(j) = (uint16_t)(j % 65536);
    Data_fedch_View.invthick_(j) = (float)(j * 1.0);
  }

  for (uint32_t j = 0; j < Data_strip_size; j++) {
    Data_strip_View.noise_(j) = (uint16_t)(j % 65536);
  }

  for (uint32_t j = 0; j < Data_apv_size; j++) {
    Data_apv_View.gain_(j) = (float)(j * -1.0f);
  }

  // Assert
  for (uint32_t j = 0; j < DetToFeds_size; j++) {
    assert(sStripCond_DetToFeds->detid_(j) == j * 2);
    assert(sStripCond_DetToFeds->ipair_(j) == (uint16_t)((j) % 65536));
    assert(sStripCond_DetToFeds->fedid_(j) == (uint16_t)((j + 1) % 65536));
    assert(sStripCond_DetToFeds->fedch_(j) == (uint8_t)(j % 256));
  }

  for (uint32_t j = 0; j < Data_fedch_size; j++) {
    assert(Data_fedch_View.detID_(j) == (uint32_t)(j));
    assert(Data_fedch_View.iPair_(j) == (uint16_t)(j % 65536));
    assert(Data_fedch_View.invthick_(j) == (float)(j * 1.0));
  }

  for (uint32_t j = 0; j < Data_strip_size; j++) {
    assert(Data_strip_View.noise_(j) == (uint16_t)(j % 65536));
  }

  for (uint32_t j = 0; j < Data_apv_size; j++) {
    assert(Data_apv_View.gain_(j) == (float)(j * -1.0f));
  }
  return 0;
}
