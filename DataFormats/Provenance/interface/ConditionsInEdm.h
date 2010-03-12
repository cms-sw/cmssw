
namespace edm{

class ConditionsInLumiBlock{
  public:
    boost::uint16_t beamMomentum;
    boost::uint32_t totalIntensityBeam1;
    boost::uint32_t totalIntensityBeam2;

    bool mergeProduct(ConditionsInLumiBlock const& newThing){
      if (beamMomentum != newThing.beamMomentum)
	return false;
      else return true;
    }
  };

  class ConditionsInRunBlock{
  public:
    boost::uint16_t beamMode;
    //    boost::uint16_t particleTypeBeam1;
    //    boost::uint16_t particleTypeBeam2;
    boost::uint32_t lhcFillNumber;
    float BStartCurrent;
    float BStopCurrent;
    float BAvgCurrent;
    bool mergeProduct(ConditionsInRunBlock const& newThing){
      if (lhcFillNumber != newThing.lhcFillNumber)
	  return false;
      else return true;
    }
  };

  
  class ConditionsInEventBlock{
  public:
    boost::uint16_t bstMasterStatus;
    boost::uint32_t turnCountNumber;
  };
}
