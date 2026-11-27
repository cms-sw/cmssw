#ifndef RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
#define RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the rechits of one HGCAL component with CLUEstering. The CLUEstering clusterer keeps its tile and seed
  // buffers from one event to the next, so one object must be used by one stream only, always with the same queue
  // (see stream::FixedQueueEDProducer).
  class HGCalCLUEsteringAlgoWrapper {
  public:
    // The scintillator rechits cluster in (eta, phi), with a periodic phi in [0, 2 pi)
    HGCalCLUEsteringAlgoWrapper(float dc, float kappa, float outlierDeltaFactor, bool isScintillator);
    ~HGCalCLUEsteringAlgoWrapper();
    HGCalCLUEsteringAlgoWrapper(const HGCalCLUEsteringAlgoWrapper&) = delete;
    HGCalCLUEsteringAlgoWrapper& operator=(const HGCalCLUEsteringAlgoWrapper&) = delete;

    // The ranges of the two coordinates for the tiles. Without them, CLUEstering computes them from the rechits of
    // every event. The tiles set the order in which the densities are summed, so the ranges can change the rounding.
    void setExtremes(std::pair<float, float> dim1, std::pair<float, float> dim2);

    // Clusters the rechits of each layer (one batch per layer, of batchItemSizes rechits): writes the layer cluster
    // and the seed flag of each rechit, and the number of layer clusters
    void run(Queue& queue,
             uint32_t nRecHits,
             std::span<const uint32_t> batchItemSizes,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering);

  private:
    struct Clusterer;
    std::unique_ptr<Clusterer> clusterer_;
    const bool isScintillator_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
