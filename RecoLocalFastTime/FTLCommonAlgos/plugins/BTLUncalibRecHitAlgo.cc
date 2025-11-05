#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/BTLRecHitsErrorEstimatorIM.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

class BTLUncalibRecHitAlgo : public BTLUncalibratedRecHitAlgoBase {
public:
  /// Constructor
  BTLUncalibRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : MTDUncalibratedRecHitAlgoBase<BTLDataFrame>(conf, sumes),
        invLightSpeedLYSO_(conf.getParameter<double>("invLightSpeedLYSO")),
        c_LYSO_(1. / invLightSpeedLYSO_),
        npeToADC_(conf.getParameter<std::vector<double>>("npeToADC")),
        npePerMeV_(conf.getParameter<double>("npePerMeV")),
        invADCPerMeV_(1. / (npeToADC_[1] * npePerMeV_)),
        tdc_to_ns_(conf.getParameter<double>("tdcLSB_ns")),
        timeError_(conf.getParameter<std::string>("timeResolutionInNs")),
        timeWalkCorr_(conf.getParameter<std::string>("timeWalkCorrection")) {}

  /// Destructor
  ~BTLUncalibRecHitAlgo() override {}

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const BTLDataFrame& dataFrame) const final;

private:
  const double invLightSpeedLYSO_;
  const double c_LYSO_;
  const std::vector<double> npeToADC_;
  const double npePerMeV_;
  const double invADCPerMeV_;
  const double tdc_to_ns_;
  const reco::FormulaEvaluator timeError_;
  const reco::FormulaEvaluator timeWalkCorr_;
};

FTLUncalibratedRecHit BTLUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame) const {
  // The reconstructed amplitudes and times of the right and left hits are saved in a std::pair
  std::pair<double, double> amplitude(0., 0.);
  std::pair<double, double> time(0., 0.);

  unsigned char flag = 0;

  const auto& sampleRight = dataFrame.sample(0);
  const auto& sampleLeft = dataFrame.sample(1);

  double nHits = 0.;

  LogDebug("BTLUncalibRecHit") << "Original input time t1, t2 " << double(sampleRight.toa()) * tdc_to_ns_ << ", "
                               << double(sampleLeft.toa()) * tdc_to_ns_ << std::endl;

  // --- Reconstruct amplitude and time of the crystal's right channel
  if (sampleRight.data() > 0) {
    // Correct the time of the right SiPM for the time-walk
    amplitude.first = double(sampleRight.data());
    time.first = double(sampleRight.toa()) -
                 timeWalkCorr_.evaluate(std::array<double, 1>{{amplitude.first}}, std::array<double, 1>{{0.0}});

    // Convert ADC counts to MeV and TDC counts to ns
    amplitude.first = (double(sampleRight.data()) - npeToADC_[0]) * invADCPerMeV_;
    time.first *= tdc_to_ns_;

    flag |= 0x1;
    nHits += 1.;
  }

  // --- Reconstruct amplitude and time of the crystal's left channel
  if (sampleLeft.data() > 0) {
    // Correct the time of the left SiPM for the time-walk
    amplitude.second = double(sampleLeft.data());
    time.second = double(sampleLeft.toa()) -
                  timeWalkCorr_.evaluate(std::array<double, 1>{{amplitude.second}}, std::array<double, 1>{{0.0}});

    // Convert ADC counts to MeV and TDC counts to ns
    amplitude.second = (double(sampleLeft.data()) - npeToADC_[0]) * invADCPerMeV_;
    time.second *= tdc_to_ns_;

    flag |= (0x1 << 1);
    nHits += 1.;
  }

  // --- Calculate the error on the hit time using the provided parameterization

  const std::array<double, 1> amplitudeV = {{(amplitude.first + amplitude.second) / nHits}};
  const std::array<double, 1> emptyV = {{0.}};

  double timeError = (nHits > 0. ? timeError_.evaluate(amplitudeV, emptyV) : -1.);

  // Calculate the position
  // Distance from center of bar to hit

  double position = 0.5f * (c_LYSO_ * (time.second - time.first));
  double positionError = BTLRecHitsErrorEstimatorIM::positionError();

  LogDebug("BTLUncalibRecHit") << "DetId: " << dataFrame.id().rawId() << " x position = " << position << " +/- "
                               << positionError;
  LogDebug("BTLUncalibRecHit") << "ADC+: set the charge to: (" << amplitude.first << ", " << amplitude.second << ")  ("
                               << sampleRight.data() << ", " << sampleLeft.data() << ") " << invADCPerMeV_ << ' '
                               << std::endl;
  LogDebug("BTLUncalibRecHit") << "TDC+: set the time to: (" << time.first << ", " << time.second << ")  ("
                               << sampleRight.toa() << ", " << sampleLeft.toa() << ") " << tdc_to_ns_ << ' '
                               << std::endl;

  return FTLUncalibratedRecHit(
      dataFrame.id(), dataFrame.row(), dataFrame.column(), amplitude, time, timeError, position, positionError, flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(BTLUncalibratedRecHitAlgoFactory, BTLUncalibRecHitAlgo, "BTLUncalibRecHitAlgo");
