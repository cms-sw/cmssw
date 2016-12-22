#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLRecHitAlgoBase.h"

class FTLSimpleRecHitAlgo : public FTLRecHitAlgoBase {
 public:
  /// Constructor
  FTLSimpleRecHitAlgo( const edm::ParameterSet& conf,
                       edm::ConsumesCollector& sumes ) : 
    FTLRecHitAlgoBase( conf, sumes ),
    thresholdToKeep_( conf.getParameter<double>("thresholdToKeep") ),
    calibration_( conf.getParameter<double>("calibrationConstant") ) { }

  /// Destructor
  virtual ~FTLSimpleRecHitAlgo() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) override final {}
  virtual void getEventSetup(const edm::EventSetup&) override final {}

  /// make the rec hit
  virtual FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const override final;

 private:  
  double thresholdToKeep_, calibration_;
};

FTLRecHit 
FTLSimpleRecHitAlgo::makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const { 
  
  float energy = uRecHit.amplitude() * calibration_;
  float time   = uRecHit.time();
  float timeError = uRecHit.timeError();
  
  FTLRecHit rh( uRecHit.id(), energy, time, timeError );
    
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
DEFINE_EDM_PLUGIN( FTLRecHitAlgoFactory, FTLSimpleRecHitAlgo, "FTLSimpleRecHitAlgo" );
