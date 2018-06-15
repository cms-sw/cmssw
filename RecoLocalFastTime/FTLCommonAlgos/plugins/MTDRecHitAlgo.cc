#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"

class MTDRecHitAlgo : public MTDRecHitAlgoBase {
 public:
  /// Constructor
  MTDRecHitAlgo( const edm::ParameterSet& conf,
                       edm::ConsumesCollector& sumes ) : 
    MTDRecHitAlgoBase( conf, sumes )
  { }

  /// Destructor
  ~MTDRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags,
		       const double& calibration, const double& thresholdToKeep) const final;

};


FTLRecHit 
MTDRecHitAlgo::makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags,
			  const double& calibration, const double& thresholdToKeep) const {

  float energy = uRecHit.amplitude() * calibration;
  float time   = uRecHit.time();
  float timeError = uRecHit.timeError();
  
  FTLRecHit rh( uRecHit.id(), energy, time, timeError );
    
  // Now fill flags
  // all rechits from the digitizer are "good" at present
  if( energy > thresholdToKeep ) {
    flags = FTLRecHit::kGood;
    rh.setFlag(flags);    
  } else {
    flags = FTLRecHit::kKilled;
    rh.setFlag(flags);
  }

  return rh;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( MTDRecHitAlgoFactory, MTDRecHitAlgo, "MTDRecHitAlgo" );
