#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDUncalibRecHitAlgo : public MTDUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  MTDUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
  MTDUncalibratedRecHitAlgoBase( conf, sumes ) { 
    uint32_t nBits     = conf.getParameter<uint32_t>("adcNbits");
    double saturation  = conf.getParameter<double>("adcSaturation");
    adcLSB_            = saturation/(1<<nBits);    
    
    toaLSBToNS_        = conf.getParameter<double>("toaLSB_ns");

    timeError_ = conf.getParameter<double>("timeResolutionInNs");
  }

  /// Destructor
  ~MTDUncalibRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const BTLDataFrame& dataFrame ) const final;
  FTLUncalibratedRecHit makeRecHit(const ETLDataFrame& dataFrame ) const final;

 private:  
  double adcLSB_, toaLSBToNS_, timeError_;
};

FTLUncalibratedRecHit 
MTDUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * adcLSB_;
  double time    = double(sample.toa()) * toaLSBToNS_;

  unsigned char flag = 0;
  
  LogDebug("MTDUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
                                     << ' ' << adcLSB_ << ' ' << std::endl;    
  LogDebug("MTDUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
                                     << ' ' << toaLSBToNS_ << ' ' << std::endl;    
  LogDebug("MTDUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, timeError_, flag);
}
FTLUncalibratedRecHit 
MTDUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * adcLSB_;
  double time    = double(sample.toa()) * toaLSBToNS_;
  unsigned char flag = 0;
  
  LogDebug("MTDUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
                                     << ' ' << adcLSB_ << ' ' << std::endl;    
  LogDebug("MTDUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
                                     << ' ' << toaLSBToNS_ << ' ' << std::endl;    
  LogDebug("MTDUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( MTDUncalibratedRecHitAlgoFactory, MTDUncalibRecHitAlgo, "MTDUncalibRecHitAlgo" );
