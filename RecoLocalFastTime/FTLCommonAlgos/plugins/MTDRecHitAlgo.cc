#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"

class MTDRecHitAlgo : public MTDRecHitAlgoBase {
 public:
  /// Constructor
  MTDRecHitAlgo( const edm::ParameterSet& conf,
                       edm::ConsumesCollector& sumes ) : 
    MTDRecHitAlgoBase( conf, sumes ),
    BTLthresholdToKeep_( conf.getParameter<double>("BTLthresholdToKeep") ),
    BTLcalibration_( conf.getParameter<double>("BTLcalibrationConstant") ),
    ETLthresholdToKeep_( conf.getParameter<double>("ETLthresholdToKeep") ),
    ETLcalibration_( conf.getParameter<double>("ETLcalibrationConstant") ) { }

  /// Destructor
  ~MTDRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLRecHit makeBTLRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const final;
  FTLRecHit makeETLRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const final;

 private:  
  double BTLthresholdToKeep_, BTLcalibration_;
  double ETLthresholdToKeep_, ETLcalibration_;
};

FTLRecHit 
MTDRecHitAlgo::makeBTLRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const { 
  
  float energy = uRecHit.amplitude() * BTLcalibration_;
  float time   = uRecHit.time();
  float timeError = uRecHit.timeError();
  
  FTLRecHit rh( uRecHit.id(), energy, time, timeError );
    
  // Now fill flags
  // all rechits from the digitizer are "good" at present
  if( energy > BTLthresholdToKeep_ ) {
    flags = FTLRecHit::kGood;
    rh.setFlag(flags);    
  } else {
    flags = FTLRecHit::kKilled;
    rh.setFlag(flags);
  }

  return rh;
}

FTLRecHit 
MTDRecHitAlgo::makeETLRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags ) const { 
  
  float energy = uRecHit.amplitude() * ETLcalibration_;
  float time   = uRecHit.time();
  float timeError = uRecHit.timeError();
  
  FTLRecHit rh( uRecHit.id(), energy, time, timeError );
    
  // Now fill flags
  // all rechits from the digitizer are "good" at present
  if( energy > ETLthresholdToKeep_ ) {
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
