#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"

class MTDRecHitAlgo : public MTDRecHitAlgoBase {
 public:
  /// Constructor
  MTDRecHitAlgo( const edm::ParameterSet& conf,
                       edm::ConsumesCollector& sumes ) : 
    MTDRecHitAlgoBase( conf, sumes ),
    thresholdToKeep_( conf.getParameter<double>("thresholdToKeep") ),
    calibration_( conf.getParameter<double>("calibrationConstant") ) { }

  /// Destructor
  ~MTDRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const final;

 private:  
  double thresholdToKeep_, calibration_;
};


FTLRecHit 
MTDRecHitAlgo::makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags) const {

  float energy = uRecHit.amplitude().first * calibration_;
  float time   = uRecHit.time().first;
  float timeError = uRecHit.timeError();

  // --- In the case of BTL bar geometry left and right informations are combined:
  if ( uRecHit.amplitude().second > 0. ) {

    energy += uRecHit.amplitude().second * calibration_;
    time = 0.5*(uRecHit.time().first + uRecHit.time().second);

  }

  FTLRecHit rh( uRecHit.id(), uRecHit.row(), uRecHit.column(), energy, time, timeError );
    
  // Now fill flags
  // all rechits from the digitizer are "good" at present
  if( energy > thresholdToKeep_ ) {
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
