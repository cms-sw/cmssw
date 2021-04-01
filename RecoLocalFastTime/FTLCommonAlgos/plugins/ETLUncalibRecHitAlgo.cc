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
        timeError_(conf.getParameter<std::string>("timeResolutionInNs")) {}

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
};

FTLUncalibratedRecHit ETLUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame) const {
  constexpr int iSample = 2;  //only in-time sample
  const auto& sample = dataFrame.sample(iSample);

  const std::array<double, 1> amplitudeV = {{double(sample.data()) * adcLSB_}};
  // NB: Here amplitudeV is defined as an array in order to be used
  //     below as an input to FormulaEvaluator::evaluate.
  double time = double(sample.toa()) * toaLSBToNS_ - tofDelay_;
  unsigned char flag = 0;

  LogDebug("ETLUncalibRecHit") << "ADC+: set the charge to: " << amplitudeV[0] << ' ' << sample.data() << ' ' << adcLSB_
                               << ' ' << std::endl;
  LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() << ' ' << toaLSBToNS_ << ' '
                               << std::endl;
  LogDebug("ETLUncalibRecHit") << "Final uncalibrated amplitude : " << amplitudeV[0] << std::endl;

  const std::array<double, 1> emptyV = {{0.}};
  double timeError = timeError_.evaluate(amplitudeV, emptyV);

  return FTLUncalibratedRecHit(dataFrame.id(),
                               dataFrame.row(),
                               dataFrame.column(),
                               {amplitudeV[0], 0.f},
                               {time, 0.f},
                               timeError,
                               -1.f,
                               -1.f,
                               flag);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(ETLUncalibratedRecHitAlgoFactory, ETLUncalibRecHitAlgo, "ETLUncalibRecHitAlgo");
