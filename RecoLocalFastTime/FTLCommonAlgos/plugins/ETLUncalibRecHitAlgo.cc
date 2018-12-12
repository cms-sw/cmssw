#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ETLUncalibRecHitAlgo : public ETLUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  ETLUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
    MTDUncalibratedRecHitAlgoBase<ETLDataFrame>( conf, sumes ),
    adcNBits_( conf.getParameter<uint32_t>("adcNbits") ),
    adcSaturation_( conf.getParameter<double>("adcSaturation") ),
    adcLSB_( adcSaturation_/(1<<adcNBits_) ),
    toaLSBToNS_( conf.getParameter<double>("toaLSB_ns") ),
    tofDelay_( conf.getParameter<double>("tofDelay") ),
    timeError_( conf.getParameter<double>("timeResolutionInNs") )
  { }

  /// Destructor
  ~ETLUncalibRecHitAlgo() override { }

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const ETLDataFrame& dataFrame ) const final;

 private:  
  
  const uint32_t adcNBits_;
  const double adcSaturation_;
  const double adcLSB_;
  const double toaLSBToNS_;
  const double tofDelay_;
  const double timeError_;

};

FTLUncalibratedRecHit 
ETLUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * adcLSB_;
  double time      = double(sample.toa()) * toaLSBToNS_ - tofDelay_;
  unsigned char flag = 0;

  LogDebug("ETLUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
			       << ' ' << adcLSB_ << ' ' << std::endl;
  LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
			       << ' ' << toaLSBToNS_ << ' ' << std::endl;
  LogDebug("ETLUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), dataFrame.row(), dataFrame.column(),
				{amplitude, 0.f}, {time, 0.f}, timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( ETLUncalibratedRecHitAlgoFactory, ETLUncalibRecHitAlgo, "ETLUncalibRecHitAlgo" );
