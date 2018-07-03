#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDUncalibRecHitAlgo : public MTDUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  MTDUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
  MTDUncalibratedRecHitAlgoBase( conf, sumes ) { 
    uint32_t BTL_nBits     = conf.getParameter<uint32_t>("BTLadcNbits");
    double BTL_saturation  = conf.getParameter<double>("BTLadcSaturation");
    BTL_adcLSB_            = BTL_saturation/(1<<BTL_nBits);
    BTL_toaLSBToNS_        = conf.getParameter<double>("BTLtoaLSB_ns");
    BTL_timeError_         = conf.getParameter<double>("BTLtimeResolutionInNs");

    uint32_t ETL_nBits     = conf.getParameter<uint32_t>("ETLadcNbits");
    double ETL_saturation  = conf.getParameter<double>("ETLadcSaturation");
    ETL_adcLSB_            = ETL_saturation/(1<<ETL_nBits);
    ETL_toaLSBToNS_        = conf.getParameter<double>("ETLtoaLSB_ns");
    ETL_timeError_         = conf.getParameter<double>("ETLtimeResolutionInNs");
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
  double BTL_adcLSB_, BTL_toaLSBToNS_, BTL_timeError_;
  double ETL_adcLSB_, ETL_toaLSBToNS_, ETL_timeError_;
};

FTLUncalibratedRecHit 
MTDUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * BTL_adcLSB_;
  double time      = double(sample.toa()) * BTL_toaLSBToNS_;

  unsigned char flag = 0;
  
  LogDebug("MTDUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
                                     << ' ' << BTL_adcLSB_ << ' ' << std::endl;
  LogDebug("MTDUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
                                     << ' ' << BTL_toaLSBToNS_ << ' ' << std::endl;
  LogDebug("MTDUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, BTL_timeError_, flag);
}
FTLUncalibratedRecHit 
MTDUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame ) const { 
  constexpr int iSample=2; //only in-time sample
  const auto& sample = dataFrame.sample(iSample);
  
  double amplitude = double(sample.data()) * ETL_adcLSB_;
  double time      = double(sample.toa()) * ETL_toaLSBToNS_;
  unsigned char flag = 0;

  LogDebug("MTDUncalibRecHit") << "ADC+: set the charge to: " << amplitude << ' ' << sample.data() 
                                     << ' ' << ETL_adcLSB_ << ' ' << std::endl;
  LogDebug("MTDUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() 
                                     << ' ' << ETL_toaLSBToNS_ << ' ' << std::endl;
  LogDebug("MTDUncalibRecHit") << "Final uncalibrated amplitude : " << amplitude << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), amplitude, time, ETL_timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( MTDUncalibratedRecHitAlgoFactory, MTDUncalibRecHitAlgo, "MTDUncalibRecHitAlgo" );
