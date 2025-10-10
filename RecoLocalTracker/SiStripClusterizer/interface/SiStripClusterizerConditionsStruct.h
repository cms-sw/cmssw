#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h

#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

namespace sistrip {
  // The underscore, such as `qualityOk_`, indicates the array is indexed with the channel index.
  // This is a mapping between (fedID, fedCh) and an integer:
  // channelIndex = (fedId-FED_ID_MIN) * FEDCH_PER_FED + fedCh;
  struct DetToFeds {
    std::array<bool, NUMBER_OF_FEDS * FEDCH_PER_FED> qualityOk_;
  };

  struct Data {
    std::array<uint32_t, NUMBER_OF_FEDS * FEDCH_PER_FED> detID_;
    std::array<float, NUMBER_OF_FEDS * FEDCH_PER_FED> invthick_;
    std::array<float, NUMBER_OF_FEDS * FEDCH_PER_FED * APVS_PER_FEDCH> gain_;
    std::array<uint16_t, NUMBER_OF_FEDS * FEDCH_PER_FED> iPair_;
    std::array<uint16_t, NUMBER_OF_FEDS * FEDCH_PER_FED * STRIPS_PER_FEDCH> noise_;
  };
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsStruct_h
