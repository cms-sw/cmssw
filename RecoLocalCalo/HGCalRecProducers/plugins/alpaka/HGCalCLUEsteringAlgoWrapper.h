#ifndef RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
#define RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h

#include <cstdint>
#include <span>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalCLUEsteringAlgoWrapper {
  public:
    void run(Queue& queue,
             const unsigned int size,
             const float dc,
             const float kappa,
             const float outlierDeltaFactor,
             const bool isScintillator,
             std::span<const uint32_t> batchItemSizes,
             const HGCalSoARecHitsDeviceCollection::ConstView inputs,
             HGCalSoARecHitsExtraDeviceCollection::View outputs) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
