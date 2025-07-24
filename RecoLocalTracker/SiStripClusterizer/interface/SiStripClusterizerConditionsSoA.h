#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsSoA_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sistrip {
  // The underscore, such as `qualityOk_`, indicates the array is indexed with the channel index.
  // This is a mapping between (fedID, fedCh) and an integer:
  // channelIndex = (fedId-FED_ID_MIN) * FEDCH_PER_FED + fedCh;
  GENERATE_SOA_LAYOUT(SiStripClusterizerConditionsDetToFedsSoALayout, SOA_COLUMN(bool, qualityOk_))

  GENERATE_SOA_LAYOUT(SiStripClusterizerConditionsData_fedch_SoALayout,
                      SOA_COLUMN(uint32_t, detID_),
                      SOA_COLUMN(uint16_t, iPair_),
                      SOA_COLUMN(float, invthick_))

  GENERATE_SOA_LAYOUT(SiStripClusterizerConditionsData_strip_SoALayout, SOA_COLUMN(uint16_t, noise_))

  GENERATE_SOA_LAYOUT(SiStripClusterizerConditionsData_apv_SoALayout, SOA_COLUMN(float, gain_))

  using SiStripClusterizerConditionsDetToFedsSoA = SiStripClusterizerConditionsDetToFedsSoALayout<>;
  using SiStripClusterizerConditionsDetToFedsView = SiStripClusterizerConditionsDetToFedsSoA::View;
  using SiStripClusterizerConditionsDetToFedsConstView = SiStripClusterizerConditionsDetToFedsSoA::ConstView;

  using SiStripClusterizerConditionsData_fedchSoA = SiStripClusterizerConditionsData_fedch_SoALayout<>;
  using SiStripClusterizerConditionsData_fedchView = SiStripClusterizerConditionsData_fedchSoA::View;
  using SiStripClusterizerConditionsData_fedchConstView = SiStripClusterizerConditionsData_fedchSoA::ConstView;

  using SiStripClusterizerConditionsData_stripSoA = SiStripClusterizerConditionsData_strip_SoALayout<>;
  using SiStripClusterizerConditionsData_stripView = SiStripClusterizerConditionsData_stripSoA::View;
  using SiStripClusterizerConditionsData_stripConstView = SiStripClusterizerConditionsData_stripSoA::ConstView;

  using SiStripClusterizerConditionsData_apvSoA = SiStripClusterizerConditionsData_apv_SoALayout<>;
  using SiStripClusterizerConditionsData_apvView = SiStripClusterizerConditionsData_apvSoA::View;
  using SiStripClusterizerConditionsData_apvConstView = SiStripClusterizerConditionsData_apvSoA::ConstView;
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsSoA_h
