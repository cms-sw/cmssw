#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

class ETLUncalibRecHitAlgo : public ETLUncalibratedRecHitAlgoBase {
public:
  /// Constructor
  ETLUncalibRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : MTDUncalibratedRecHitAlgoBase<ETLDataFrame>(conf, sumes),
        adcNBits_(conf.getParameter<uint32_t>("adcNbits")),
        adcSaturation_(conf.getParameter<double>("adcSaturation")),
        adcLSB_(adcSaturation_ / (1 << adcNBits_)),
        toaLSBToNS_(conf.getParameter<double>("toaLSB_ns")),
        tofDelay_(conf.getParameter<double>("tofDelay")),
        timeError_(conf.getParameter<std::string>("timeResolutionInNs")),
        timeCorr_p0_(conf.getParameter<double>("timeCorr_p0")),
        timeCorr_p1_(conf.getParameter<double>("timeCorr_p1")),
        timeCorr_p2_(conf.getParameter<double>("timeCorr_p2")),
        timeCorr_p3_(conf.getParameter<double>("timeCorr_p3")) {}
  /// Destructor
  ~ETLUncalibRecHitAlgo() override {}

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final {}

  /// make the rec hit
  FTLUncalibratedRecHit makeRecHit(const ETLDataFrame& dataFrame) const final;

private:
  const uint32_t adcNBits_;
  const double adcSaturation_;
  const double adcLSB_;
  const double toaLSBToNS_;
  const double tofDelay_;
  const reco::FormulaEvaluator timeError_;
  const double timeCorr_p0_;
  const double timeCorr_p1_;
  const double timeCorr_p2_;
  const double timeCorr_p3_;
};

FTLUncalibratedRecHit ETLUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame) const {
  constexpr int iSample = 2;  //only in-time sample
  const auto& sample = dataFrame.sample(iSample);

  double time = double(sample.toa()) * toaLSBToNS_ - tofDelay_;
  double time_over_threshold = double(sample.tot()) * toaLSBToNS_;
  const std::array<double, 1> time_over_threshold_V = {{time_over_threshold}};

  unsigned char flag = 0;

  LogDebug("ETLUncalibRecHit") << "ADC+: set the charge to: " << time_over_threshold << ' ' << sample.tot() << ' '
                               << toaLSBToNS_;

  if (time_over_threshold == 0) {
    LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() << ' ' << toaLSBToNS_;

  } else {
    // Time-walk correction for toa
    double timeWalkCorr = timeCorr_p0_ + timeCorr_p1_ * time_over_threshold +
                          timeCorr_p2_ * time_over_threshold * time_over_threshold +
                          timeCorr_p3_ * time_over_threshold * time_over_threshold * time_over_threshold;

    time -= timeWalkCorr;

    LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() << ' ' << toaLSBToNS_
                                 << " .Timewalk correction: " << timeWalkCorr;
  }

  LogDebug("ETLUncalibRecHit") << "Final uncalibrated time_over_threshold: " << time_over_threshold;

  const std::array<double, 1> emptyV = {{0.}};

  double timeError = timeError_.evaluate(time_over_threshold_V, emptyV);

  return FTLUncalibratedRecHit(dataFrame.id(),
                               dataFrame.row(),
                               dataFrame.column(),
                               {time_over_threshold_V[0], 0.f},
                               {time, 0.f},
                               timeError,
                               -1.f,
                               -1.f,
                               flag);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(ETLUncalibratedRecHitAlgoFactory, ETLUncalibRecHitAlgo, "ETLUncalibRecHitAlgo");
