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
  using LayerTriplet = std::tuple<SeedingLayerSetsHits::LayerIndex,
                                  SeedingLayerSetsHits::LayerIndex,
                                  SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  ////////////////////

  class LayerTripletHitIndex {
  public:
    LayerTripletHitIndex(const SeedingLayerSetsHits::SeedingLayerSet& layerPair, const SeedingLayerSetsHits::SeedingLayer& thirdLayer,
                         size_t begin, LayerHitMapCache&& cache):
      layerTriplet_(layerPair[0].index(), layerPair[1].index(), thirdLayer.index()),
      hitsBegin_(begin),
      cache_(std::move(cache))
    {}

  private:
    LayerTriplet layerTriplet_;
    size_t hitsBegin_;
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
    std::swap(layerTriplets_, rh.layerTriplets_);
    std::swap(hitTriplets_, rh.hitTriplets_);
  }

  void reserve(size_t nregions, size_t nlayersets, size_t ntriplets) {
    regions_.reserve(nregions);
    layerTriplets_.reserve(nregions*nlayersets);
    hitTriplets_.reserve(ntriplets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerTriplets_.shrink_to_fit();
    hitTriplets_.shrink_to_fit();
  }

  void beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, layerTriplets_.size());
  }

  void addTriplets(const SeedingLayerSetsHits::SeedingLayerSet& layerPair, const SeedingLayerSetsHits::SeedingLayer& thirdLayer,
                   OrderedHitTriplets::iterator hitTripletsBegin, OrderedHitTriplets::iterator hitTripletsEnd,
                   LayerHitMapCache&& cache) {
    layerTriplets_.emplace_back(layerPair, thirdLayer, std::distance(hitTripletsBegin, hitTripletsEnd), std::move(cache));
    std::move(hitTripletsBegin, hitTripletsEnd, std::back_inserter(hitTriplets_)); // probably not much different from std::copy as we're just moving pointers...
  }

private:
  const SeedingLayerSetsHits *seedingLayers_;

  std::vector<RegionIndex> regions_;
  std::vector<LayerTripletHitIndex> layerTriplets_;
  std::vector<OrderedHitTriplet> hitTriplets_;
};

#endif
