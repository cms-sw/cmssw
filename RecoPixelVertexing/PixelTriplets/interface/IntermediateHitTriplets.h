#ifndef RecoPixelVertexing_PixelTriplets_IntermediateHitTriplets_h
#define RecoPixelVertexing_PixelTriplets_IntermediateHitTriplets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"

/**
 * Simple container of temporary information delivered from hit triplet
 * generator to hit quadruplet generator via edm::Event.
 */
class IntermediateHitTriplets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex,
                               SeedingLayerSetsHits::LayerIndex>;
  using LayerTriplet = std::tuple<SeedingLayerSetsHits::LayerIndex,
                                  SeedingLayerSetsHits::LayerIndex,
                                  SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  ////////////////////

  class ThirdLayer {
  public:
    ThirdLayer(const SeedingLayerSetsHits::SeedingLayer& thirdLayer, size_t hitsBegin):
      thirdLayer_(thirdLayer.index()), hitsBegin_(hitsBegin)
    {}

  private:
    SeedingLayerSetsHits::LayerIndex thirdLayer_;
    size_t hitsBegin_;
  };

  ////////////////////

  class LayerPairAndLayers {
  public:
    LayerPairAndLayers(const SeedingLayerSetsHits::SeedingLayerSet& layerPair,
                       size_t thirdLayersBegin, LayerHitMapCache&& cache):
      layerPair_(layerPair[0].index(), layerPair[1].index()),
      thirdLayersBegin_(thirdLayersBegin),
      cache_(std::move(cache))
    {}

  private:
    LayerPair layerPair_;
    size_t thirdLayersBegin_;
    LayerHitMapCache cache_;
  };

  ////////////////////

  IntermediateHitTriplets(): seedingLayers_(nullptr) {}
  explicit IntermediateHitTriplets(const SeedingLayerSetsHits *seedingLayers): seedingLayers_(seedingLayers) {}
  IntermediateHitTriplets(const IntermediateHitTriplets& rh); // only to make ROOT dictionary generation happy
  ~IntermediateHitTriplets() = default;

  void swap(IntermediateHitTriplets& rh) {
    std::swap(seedingLayers_, rh.seedingLayers_);
    std::swap(regions_, rh.regions_);
    std::swap(layerPairAndLayers_, rh.layerPairAndLayers_);
    std::swap(thirdLayers_, rh.thirdLayers_);
    std::swap(hitTriplets_, rh.hitTriplets_);
  }

  void reserve(size_t nregions, size_t nlayersets, size_t ntriplets) {
    regions_.reserve(nregions);
    layerPairAndLayers_.reserve(nregions*nlayersets);
    thirdLayers_.reserve(nregions*nlayersets);
    hitTriplets_.reserve(ntriplets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerPairAndLayers_.shrink_to_fit();
    thirdLayers_.shrink_to_fit();
    hitTriplets_.shrink_to_fit();
  }

  void beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, layerPairAndLayers_.size());
  }

  void beginPair(const SeedingLayerSetsHits::SeedingLayerSet& layerPair, LayerHitMapCache&& cache) {
    layerPairAndLayers_.emplace_back(layerPair, thirdLayers_.size(), std::move(cache));
  };

  void addTriplets(const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                   const OrderedHitTriplets& triplets,
                   const std::vector<int>& thirdLayerIndex,
                   const std::vector<size_t>& permutations) {
    assert(triplets.size() == thirdLayerIndex.size());
    assert(triplets.size() == permutations.size());

    int prevLayer = -1;
    for(size_t i=0, size=permutations.size(); i<size; ++i) {
      // We go through the 'triplets' in the order defined by
      // 'permutations', which is sorted such that we first go through
      // triplets from (3rd) layer 0, then layer 1 and so on.
      const size_t realIndex = permutations[i];

      const int layer = thirdLayerIndex[realIndex];
      if(layer != prevLayer) {
        prevLayer = layer;
        thirdLayers_.emplace_back(thirdLayers[layer], hitTriplets_.size());
      }

      hitTriplets_.emplace_back(triplets[realIndex]);
    }
  }

private:
  const SeedingLayerSetsHits *seedingLayers_;

  std::vector<RegionIndex> regions_;
  std::vector<LayerPairAndLayers> layerPairAndLayers_;
  std::vector<ThirdLayer> thirdLayers_;
  std::vector<OrderedHitTriplet> hitTriplets_;
};

#endif
