#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"

class ETLUncalibRecHitAlgo : public ETLUncalibratedRecHitAlgoBase {
public:
  /// Constructor
  ETLUncalibRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : MTDUncalibratedRecHitAlgoBase<ETLDataFrame>(conf, sumes),
        adcNBits_(conf.getParameter<uint32_t>("adcNbits")),
        adcSaturation_(conf.getParameter<double>("adcSaturation")),
        adcLSB_(adcSaturation_ / (1 << adcNBits_)),
        toaLSBToNS_(conf.getParameter<double>("toaLSB_ns")),
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

  /// Fill parameter descriptions for validation
  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  const uint32_t adcNBits_;
  const double adcSaturation_;
  const double adcLSB_;
  const double toaLSBToNS_;
  const reco::FormulaEvaluator timeError_;
  const double timeCorr_p0_;
  const double timeCorr_p1_;
  const double timeCorr_p2_;
  const double timeCorr_p3_;
};
