#ifndef DataFormats_Common_ConditionsInEdm_h 
#define DataFormats_Common_ConditionsInEdm_h 

#include <boost/cstdint.hpp>

namespace edm {

  class ConditionsInLumiBlock {
  public:
    boost::uint16_t beamMomentum;
    boost::uint32_t totalIntensityBeam1;
    boost::uint32_t totalIntensityBeam2;

    bool mergeProduct(ConditionsInLumiBlock const& newThing) {
      return (beamMomentum == newThing.beamMomentum);
    }
  };

  class ConditionsInRunBlock {
  public:
    boost::uint16_t beamMode;
    //    boost::uint16_t particleTypeBeam1;
    //    boost::uint16_t particleTypeBeam2;
    boost::uint32_t lhcFillNumber;
    float BStartCurrent;
    float BStopCurrent;
    float BAvgCurrent;
    bool mergeProduct(ConditionsInRunBlock const& newThing) {
      return (lhcFillNumber == newThing.lhcFillNumber);
    }
  };

  
  class ConditionsInEventBlock {
  public:
    boost::uint16_t bstMasterStatus;
    boost::uint32_t turnCountNumber;
  };
}
#endif
