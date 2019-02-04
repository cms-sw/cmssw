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

  // The reconstructed amplitudes and times are saved in a std::pair
  //    BTL tile geometry (1 SiPM): only the first value of the amplitude
  //                                and time pairs is used.
  //    BTL bar geometry (2 SiPMs): both values of the amplitude and
  //                                time pairs are filled.

  std::pair<float,float> amplitude(0.,0.);
  std::pair<float,float> time(0.,0.);

  unsigned char flag = 0;

  const auto& sampleLeft  = dataFrame.sample(0);
  const auto& sampleRight = dataFrame.sample(1);
  
  if ( sampleLeft.data() > 0 ) {

    amplitude.first = float(sampleLeft.data()) * adcLSB_;
    time.first      = float(sampleLeft.toa()) * toaLSBToNS_;

    // Correct the time of the left SiPM for the time-walk
    time.first -= timeCorr_p0_*pow(amplitude.first,timeCorr_p1_) + timeCorr_p2_;
    flag |= 0x1;

  }

  // --- If available, reconstruct the amplitude and time of the second SiPM
  if ( sampleRight.data() > 0 ) {

    amplitude.second = sampleRight.data() * adcLSB_;
    time.second      = sampleRight.toa() * toaLSBToNS_;

    // Correct the time of the right SiPM for the time-walk
    time.second -= timeCorr_p0_*pow(amplitude.second,timeCorr_p1_) + timeCorr_p2_;
    flag |= (0x1 << 1);

  }

  LogDebug("BTLUncalibRecHit") << "ADC+: set the charge to: (" << amplitude.first << ", "
			       << amplitude.second << ")  ("
			       << sampleLeft.data() << ", " << sampleRight.data()
			       << "  " << adcLSB_ << ' ' << std::endl;
  LogDebug("BTLUncalibRecHit") << "TDC+: set the time to: (" << time.first << ", "
			       << time.second << ")  ("
			       << sampleLeft.toa() << ", " << sampleRight.toa()
			       << "  " << toaLSBToNS_ << ' ' << std::endl;
  
  return FTLUncalibratedRecHit( dataFrame.id(), dataFrame.row(), dataFrame.column(),
				amplitude, time, timeError_, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN( BTLUncalibratedRecHitAlgoFactory, BTLUncalibRecHitAlgo, "BTLUncalibRecHitAlgo" );
