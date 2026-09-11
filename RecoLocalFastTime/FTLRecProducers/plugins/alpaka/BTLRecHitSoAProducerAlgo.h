#ifndef RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLRecHitSoAProducerAlgo_h
#define RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLRecHitSoAProducerAlgo_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  using namespace ::btlrechit;

  struct BTLRecHitSoAProducerAlgo {
    static void fromBaseToReco(Queue& queue,
                               BTLBaseRecHitSoA::ConstView const& input,
                               BTLRecHitSoA::View& output,
                               double c_LYSO_,
                               double thresholdToKeep_,
                               double calibration_,
                               const std::array<double, 2> npeSaturationCorr_,
                               const std::array<double, 2> npeToADC_,
                               const double npePerGeV_,
                               const double timeCalibration_,
                               const std::array<double, 3> tResParams_,
                               const std::array<double, 3> twcParams_);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit

#endif  // RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLRecHitSoAProducerAlgo_h
