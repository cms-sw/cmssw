#ifndef RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLBaseRecHitSoAProducerAlgo_h
#define RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLBaseRecHitSoAProducerAlgo_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"
#include "DataFormats/FTLDigiSoA/interface/ETLDigiSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using namespace ::etlrechit;

  struct ETLBaseRecHitSoAProducerAlgo {
    static void fromDigiToBase(Queue& queue,
                               ::etldigi::ETLDigiSoA::ConstView const& input,
                               ETLBaseRecHitSoA::View& output,
                               const uint32_t adcNBits_,
                               const double adcSaturation_,
                               const double adcLSB_,
                               const double toaLSB_ns_,
                               const double tdcWindowStart_);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit

#endif  // RecoLocalFastTime_FTLCommonAlgos_plugins_alpaka_ETLBaseRecHitSoAProducerAlgo_h
