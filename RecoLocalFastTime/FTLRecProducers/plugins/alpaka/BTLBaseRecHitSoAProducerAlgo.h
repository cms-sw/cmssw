#ifndef RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLBaseRecHitSoAProducerAlgo_h
#define RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLBaseRecHitSoAProducerAlgo_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"
#include "DataFormats/FTLDigiSoA/interface/BTLDigiSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  using namespace ::btlrechit;

  struct BTLBaseRecHitSoAProducerAlgo {
    static void fromDigiToBase(Queue& queue,
                               ::btldigi::BTLDigiSoA::ConstView const& input,
                               BTLBaseRecHitSoA::View& output,
                               const uint32_t adcBitSaturation_,
                               const double tclock_,
                               const std::array<double, 4> tdcCalParams_,
                               const std::array<double, 10> qdcCalParams_);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit

#endif  // RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_BTLBaseRecHitSoAProducerAlgo_h
