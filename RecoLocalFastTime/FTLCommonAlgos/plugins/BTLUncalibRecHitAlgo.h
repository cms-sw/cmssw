#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"

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
        npeSaturationCorr_(conf.getParameter<std::vector<double>>("npeSaturationCorrection")),
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

  /// Fill parameter descriptions for validation
  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  const double invLightSpeedLYSO_;
  const double c_LYSO_;
  const std::vector<double> npeToADC_;
  const double npePerMeV_;
  const double invADCPerMeV_;
  const std::vector<double> npeSaturationCorr_;
  const double tdc_to_ns_;
  const reco::FormulaEvaluator timeError_;
  const reco::FormulaEvaluator timeWalkCorr_;
};
