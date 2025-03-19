// A minimal test to ensure that
//   - sistrip::SiStripClusterizerConditionsSoA, sistrip::SiStripClusterizerConditionsHost can be compiled
//   - sistrip::SiStripClusterizerConditionsSoA can be allocated, modified and erased (on host)
//   - view-based element access works

// #include <cstdint>
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"

int main() {
  // using SiStripClusterizerConditionsHost = PortableHostMultiCollection<SiStripClusterizerConditionsDetToFedsSoA, SiStripClusterizerConditionsData_fedch_SoA, SiStripClusterizerConditionsData_strip_SoA, SiStripClusterizerConditionsData_apv_SoA>;
  int const DetToFedsSoA_size = 10;
  int const Data_fedch_SoA_size = 50;
  int const Data_strip_SoA_size = 100;
  int const Data_apv_SoA_size = 200;
  SiStripClusterizerConditionsHost collection(
      {{DetToFedsSoA_size, Data_fedch_SoA_size, Data_strip_SoA_size, Data_apv_SoA_size}}, cms::alpakatools::host());

  collection.zeroInitialise();

  auto DetToFeds_view = collection.view();
  auto Data_fedchSoA_view = collection.view<SiStripClusterizerConditionsData_fedchSoA>();
  auto Data_stripSoA_view = collection.view<SiStripClusterizerConditionsData_stripSoA>();
  auto Data_apvSoA_view = collection.view<SiStripClusterizerConditionsData_apvSoA>();

  for (uint32_t j = 0; j < DetToFedsSoA_size; j++) {
    DetToFeds_view[j].detid_() = j * 2;
    DetToFeds_view[j].ipair_() = (uint16_t)((j) % 65536);
    DetToFeds_view[j].fedid_() = (uint16_t)((j + 1) % 65536);
    DetToFeds_view[j].fedch_() = (uint8_t)(j % 256);
  }

  for (uint32_t j = 0; j < Data_fedch_SoA_size; j++) {
    Data_fedchSoA_view[j].detID_() = (uint32_t)(j);
    Data_fedchSoA_view[j].iPair_() = (uint16_t)(j % 65536);
    Data_fedchSoA_view[j].invthick_() = (float)(j * 1.0);
  }

  for (uint32_t j = 0; j < Data_strip_SoA_size; j++) {
    Data_stripSoA_view[j].noise_() = (uint16_t)(j % 65536);
  }

  for (uint32_t j = 0; j < Data_apv_SoA_size; j++) {
    Data_apvSoA_view[j].gain_() = (float)(j * -1.0f);
  }

  // Assert
  for (uint32_t j = 0; j < DetToFedsSoA_size; j++) {
    assert(DetToFeds_view[j].detid_() == j * 2);
    assert(DetToFeds_view[j].ipair_() == (uint16_t)((j) % 65536));
    assert(DetToFeds_view[j].fedid_() == (uint16_t)((j + 1) % 65536));
    assert(DetToFeds_view[j].fedch_() == (uint8_t)(j % 256));
  }

  for (uint32_t j = 0; j < Data_fedch_SoA_size; j++) {
    assert(Data_fedchSoA_view[j].detID_() == (uint32_t)(j));
    assert(Data_fedchSoA_view[j].iPair_() == (uint16_t)(j % 65536));
    assert(Data_fedchSoA_view[j].invthick_() == (float)(j * 1.0));
  }

  for (uint32_t j = 0; j < Data_strip_SoA_size; j++) {
    assert(Data_stripSoA_view[j].noise_() == (uint16_t)(j % 65536));
  }

  for (uint32_t j = 0; j < Data_apv_SoA_size; j++) {
    assert(Data_apvSoA_view[j].gain_() == (float)(j * -1.0f));
  }
  return 0;
}
