#ifndef RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLRecHitSoAProducerAlgo_h
#define RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLRecHitSoAProducerAlgo_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using namespace ::etlrechit;

  struct ETLRecHitSoAProducerAlgo {
    static void fromBaseToReco(Queue& queue,
                               ETLBaseRecHitSoA::ConstView const& input,
                               ETLRecHitSoA::View& output,
                               double thresholdToKeep_,
                               double calibration_,
                               double timeResInNs_,
                               const double timeCorr_p0_,
                               const double timeCorr_p2_,
                               const double timeCorr_p1_,
                               const double timeCorr_p3_);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit

#endif  // RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLRecHitSoAProducerAlgo_h
