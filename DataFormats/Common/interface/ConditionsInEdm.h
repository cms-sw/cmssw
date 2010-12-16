#ifndef DataFormats_Common_ConditionsInEdm_h 
#define DataFormats_Common_ConditionsInEdm_h 

#include <boost/cstdint.hpp>

namespace edm {

  class ConditionsInLumiBlock {
  public:
    boost::uint32_t totalIntensityBeam1;
    boost::uint32_t totalIntensityBeam2;

    bool isProductEqual(ConditionsInLumiBlock const& newThing) const {
      return ((totalIntensityBeam1 == newThing.totalIntensityBeam1) &&
	      (totalIntensityBeam2 == newThing.totalIntensityBeam2));
    }
  };

  class ConditionsInRunBlock {
  public:
    boost::uint16_t beamMode;
    boost::uint16_t beamMomentum;
    //    boost::uint16_t particleTypeBeam1;
    //    boost::uint16_t particleTypeBeam2;
    boost::uint32_t lhcFillNumber;
    float BStartCurrent;
    float BStopCurrent;
    float BAvgCurrent;
    bool isProductEqual(ConditionsInRunBlock const& newThing) const {
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
