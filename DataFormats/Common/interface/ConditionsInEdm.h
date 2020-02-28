#include <cstdint>
#ifndef DataFormats_Common_ConditionsInEdm_h
#define DataFormats_Common_ConditionsInEdm_h

namespace edm {

  class ConditionsInLumiBlock {
  public:
    uint32_t totalIntensityBeam1;
    uint32_t totalIntensityBeam2;

    bool isProductEqual(ConditionsInLumiBlock const& newThing) const {
      return ((totalIntensityBeam1 == newThing.totalIntensityBeam1) &&
              (totalIntensityBeam2 == newThing.totalIntensityBeam2));
    }
  };

  class ConditionsInRunBlock {
  public:
    uint16_t beamMode;
    uint16_t beamMomentum;
    //    uint16_t particleTypeBeam1;
    //    uint16_t particleTypeBeam2;
    uint32_t lhcFillNumber;
    float BStartCurrent;
    float BStopCurrent;
    float BAvgCurrent;
    bool isProductEqual(ConditionsInRunBlock const& newThing) const {
      return (lhcFillNumber == newThing.lhcFillNumber);
    }
  };

  class ConditionsInEventBlock {
  public:
    uint16_t bstMasterStatus;
    uint32_t turnCountNumber;
  };
}  // namespace edm
#endif
