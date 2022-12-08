#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/BTLRecHitsErrorEstimatorIM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

class BTLUncalibRecHitAlgo : public BTLUncalibratedRecHitAlgoBase {
public:
  /// Constructor
  BTLUncalibRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : MTDUncalibratedRecHitAlgoBase<BTLDataFrame>(conf, sumes),
        adcNBits_(conf.getParameter<uint32_t>("adcNbits")),
        adcSaturation_(conf.getParameter<double>("adcSaturation")),
        adcLSB_(adcSaturation_ / (1 << adcNBits_)),
        toaLSBToNS_(conf.getParameter<double>("toaLSB_ns")),
        timeError_(conf.getParameter<std::string>("timeResolutionInNs")),
        timeCorr_p0_(conf.getParameter<double>("timeCorr_p0")),
        timeCorr_p1_(conf.getParameter<double>("timeCorr_p1")),
        timeCorr_p2_(conf.getParameter<double>("timeCorr_p2")),
        c_LYSO_(conf.getParameter<double>("c_LYSO")) {}

  /// Destructor
  ~BTLUncalibRecHitAlgo() override {}

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const BTLDataFrame& dataFrame) const final;

private:
  const uint32_t adcNBits_;
  const double adcSaturation_;
  const double adcLSB_;
  const double toaLSBToNS_;
  const reco::FormulaEvaluator timeError_;
  const double timeCorr_p0_;
  const double timeCorr_p1_;
  const double timeCorr_p2_;
  const double c_LYSO_;
};

FTLUncalibratedRecHit BTLUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame) const {
  // The reconstructed amplitudes and times are saved in a std::pair
  //    BTL tile geometry (1 SiPM): only the first value of the amplitude
  //                                and time pairs is used.
  //    BTL bar geometry (2 SiPMs): both values of the amplitude and
  //                                time pairs are filled.

  std::pair<float, float> amplitude(0., 0.);
  std::pair<float, float> time(0., 0.);

  unsigned char flag = 0;

  const auto& sampleRight = dataFrame.sample(0);
  const auto& sampleLeft = dataFrame.sample(1);

  double nHits = 0.;

  if (sampleRight.data() > 0) {
    amplitude.first = float(sampleRight.data()) * adcLSB_;
    time.first = float(sampleRight.toa()) * toaLSBToNS_;

    nHits += 1.;

    // Correct the time of the left SiPM for the time-walk
    time.first -= timeCorr_p0_ * pow(amplitude.first, timeCorr_p1_) + timeCorr_p2_;
    flag |= 0x1;
  }

  // --- If available, reconstruct the amplitude and time of the second SiPM
  if (sampleLeft.data() > 0) {
    amplitude.second = float(sampleLeft.data()) * adcLSB_;
    time.second = float(sampleLeft.toa()) * toaLSBToNS_;

    nHits += 1.;

    // Correct the time of the right SiPM for the time-walk
    time.second -= timeCorr_p0_ * pow(amplitude.second, timeCorr_p1_) + timeCorr_p2_;
    flag |= (0x1 << 1);
  }

  // --- Calculate the error on the hit time using the provided parameterization

  const std::array<double, 1> amplitudeV = {{(amplitude.first + amplitude.second) / nHits}};
  const std::array<double, 1> emptyV = {{0.}};

  double timeError = (nHits > 0. ? timeError_.evaluate(amplitudeV, emptyV) : -1.);

  // Calculate the position
  // Distance from center of bar to hit
  float position = 0.5f * (c_LYSO_ * (time.second - time.first));
  float positionError = BTLRecHitsErrorEstimatorIM::positionError();

  LogDebug("BTLUncalibRecHit") << "ADC+: set the charge to: (" << amplitude.first << ", " << amplitude.second << ")  ("
                               << sampleRight.data() << ", " << sampleLeft.data() << "  " << adcLSB_ << ' '
                               << std::endl;
  LogDebug("BTLUncalibRecHit") << "TDC+: set the time to: (" << time.first << ", " << time.second << ")  ("
                               << sampleRight.toa() << ", " << sampleLeft.toa() << "  " << toaLSBToNS_ << ' '
                               << std::endl;

  return FTLUncalibratedRecHit(
      dataFrame.id(), dataFrame.row(), dataFrame.column(), amplitude, time, timeError, position, positionError, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(BTLUncalibratedRecHitAlgoFactory, BTLUncalibRecHitAlgo, "BTLUncalibRecHitAlgo");
