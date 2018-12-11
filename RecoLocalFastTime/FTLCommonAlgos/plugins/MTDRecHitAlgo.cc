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

  unsigned char flagsWord = uRecHit.flags();
  float timeError = uRecHit.timeError();

  float energy = 0.;
  float time   = 0.;

  switch ( flagsWord ) {
    // BTL bar geometry with only the right SiPM information available
    case 0x2 : {

      energy = uRecHit.amplitude().second;
      time   = uRecHit.time().second;

      break ;
    }
    // BTL bar geometry with left and right SiPMs information available
    case 0x3 : {

      energy = 0.5*(uRecHit.amplitude().first + uRecHit.amplitude().second);
      time   = 0.5*(uRecHit.time().first + uRecHit.time().second);

      break ;
    }
    // ETL, BTL tile geometry, BTL bar geometry with only the left SiPM information available
    default: {

      energy = uRecHit.amplitude().first;
      time   = uRecHit.time().first;

      break ;
    }
  }

  // --- Energy calibration: for the time being this is just a conversion pC --> MeV
  energy *= calibration_;

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
