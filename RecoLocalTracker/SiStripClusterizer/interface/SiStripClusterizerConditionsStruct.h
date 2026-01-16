#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h

#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

namespace sistrip {
  // Quality, detID, invThick and iPair are indexed by the `channelIndex` map
  // Gain and noise are indexed by the stripIndex and apvIndex, respectively
  struct DetToFeds {
    std::array<bool, NUMBER_OF_FEDS * FEDCH_PER_FED> qualityOk;
  };

  struct GainNoiseCals {
    std::array<uint32_t, NUMBER_OF_FEDS * FEDCH_PER_FED> detID;
    std::array<float, NUMBER_OF_FEDS * FEDCH_PER_FED> invthick;
    std::array<float, NUMBER_OF_FEDS * FEDCH_PER_FED * APVS_PER_FEDCH> gain;
    std::array<uint16_t, NUMBER_OF_FEDS * FEDCH_PER_FED> iPair;
    std::array<uint16_t, NUMBER_OF_FEDS * FEDCH_PER_FED * STRIPS_PER_FEDCH> noise;
  };
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h
