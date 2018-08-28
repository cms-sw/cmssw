#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class BTLUncalibRecHitAlgo : public BTLUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  BTLUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
    MTDUncalibratedRecHitAlgoBase<BTLDataFrame>( conf, sumes ),
    adcNBits_( conf.getParameter<uint32_t>("adcNbits") ),
    adcSaturation_( conf.getParameter<double>("adcSaturation") ),
    adcLSB_( adcSaturation_/(1<<adcNBits_) ),
    toaLSBToNS_( conf.getParameter<double>("toaLSB_ns") ),
    timeError_( conf.getParameter<double>("timeResolutionInNs") ),
    timeCorr_p0_( conf.getParameter<double>("timeCorr_p0") ),
    timeCorr_p1_( conf.getParameter<double>("timeCorr_p1") ),
    timeCorr_p2_( conf.getParameter<double>("timeCorr_p2") )
  { }

  /// Destructor
  ~BTLUncalibRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const BTLDataFrame& dataFrame ) const final;

 private:  
  
  const uint32_t adcNBits_;
  const double adcSaturation_;
  const double adcLSB_;
  const double toaLSBToNS_;
  const double timeError_;
  const double timeCorr_p0_;
  const double timeCorr_p1_;
  const double timeCorr_p2_;

};

FTLUncalibratedRecHit 
BTLUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * adcLSB_;
  double time      = double(sample.toa()) * toaLSBToNS_;

  // --- Correct the time for the time-walk and the constant delays
  if ( amplitude > 0. )
    time -= timeCorr_p0_*pow(amplitude,timeCorr_p1_) + timeCorr_p2_;

  unsigned char flag = 0;
  
  LogDebug("BTLUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
			       << ' ' << adcLSB_ << ' ' << std::endl;
  LogDebug("BTLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
			       << ' ' << toaLSBToNS_ << ' ' << std::endl;
  LogDebug("BTLUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( BTLUncalibratedRecHitAlgoFactory, BTLUncalibRecHitAlgo, "BTLUncalibRecHitAlgo" );
