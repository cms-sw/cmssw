#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FTLSimpleUncalibRecHitAlgo : public FTLUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  FTLSimpleUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
  FTLUncalibratedRecHitAlgoBase( conf, sumes ) { 
    uint32_t nBits     = conf.getParameter<uint32_t>("adcNbits");
    double saturation  = conf.getParameter<double>("adcSaturation");
    adcLSB_            = saturation/(1<<nBits);    
    
    toaLSBToNS_        = conf.getParameter<double>("toaLSB_ns");

    timeError_ = conf.getParameter<double>("timeResolutionInNs");
  }

  /// Destructor
  virtual ~FTLSimpleUncalibRecHitAlgo() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) override final {}
  virtual void getEventSetup(const edm::EventSetup&) override final {}

  /// make the rec hit
  virtual FTLUncalibratedRecHit makeRecHit(const FTLDataFrame& dataFrame ) const override final;

 private:  
  double adcLSB_, toaLSBToNS_, timeError_;
};

FTLUncalibratedRecHit 
FTLSimpleUncalibRecHitAlgo::makeRecHit(const FTLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * adcLSB_;
  double time    = double(sample.toa()) * toaLSBToNS_;
  unsigned char flag = 0;
  
  LogDebug("FTLSimpleUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
                                     << ' ' << adcLSB_ << ' ' << std::endl;    
  LogDebug("FTLSimpleUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
                                     << ' ' << toaLSBToNS_ << ' ' << std::endl;    
  LogDebug("FTLSimpleUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( FTLUncalibratedRecHitAlgoFactory, FTLSimpleUncalibRecHitAlgo, "FTLSimpleUncalibRecHitAlgo" );
