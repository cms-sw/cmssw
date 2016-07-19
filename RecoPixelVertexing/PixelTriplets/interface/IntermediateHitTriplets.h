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
    ThirdLayer(const SeedingLayerSetsHits::SeedingLayer& thirdLayer, unsigned int hitsBegin):
      thirdLayer_(thirdLayer.index()), hitsBegin_(hitsBegin), hitsEnd_(hitsBegin)
    {}

    void setHitsEnd(unsigned int end) { hitsEnd_ = end; }

    SeedingLayerSetsHits::LayerIndex layerIndex() const { return thirdLayer_; }

    unsigned int tripletsBegin() const { return hitsBegin_; }
    unsigned int tripletsEnd() const { return hitsEnd_; }

  private:
    SeedingLayerSetsHits::LayerIndex thirdLayer_;
    unsigned int hitsBegin_;
    unsigned int hitsEnd_;
  };

  ////////////////////

  class LayerPairAndLayers {
  public:
    LayerPairAndLayers(const LayerPair& layerPair,
                       unsigned int thirdLayersBegin, LayerHitMapCache&& cache):
      layerPair_(layerPair),
      thirdLayersBegin_(thirdLayersBegin),
      thirdLayersEnd_(thirdLayersBegin),
      cache_(std::move(cache))
    {}

    void setThirdLayersEnd(unsigned int end) { thirdLayersEnd_ = end; }

    const LayerPair& layerPair() const { return layerPair_; }
    unsigned int thirdLayersBegin() const { return thirdLayersBegin_; }
    unsigned int thirdLayersEnd() const { return thirdLayersEnd_; }

    LayerHitMapCache& cache() { return cache_; }
    const LayerHitMapCache& cache() const { return cache_; }

  private:
    LayerPair layerPair_;
    unsigned int thirdLayersBegin_;
    unsigned int thirdLayersEnd_;
    // The reason for not storing layer triplets + hit triplets
    // directly is in this cache, and in my desire to try to keep
    // results unchanged during this refactoring
    LayerHitMapCache cache_;
  };

  ////////////////////

  class LayerTripletHits {
  public:
    LayerTripletHits(const IntermediateHitTriplets *hitSets,
                     const LayerPairAndLayers *layerPairAndLayers,
                     const ThirdLayer *thirdLayer):
      hitSets_(hitSets),
      layerPairAndLayers_(layerPairAndLayers),
      thirdLayer_(thirdLayer)
    {}

    using TripletRange = std::pair<std::vector<OrderedHitTriplet>::const_iterator,
                                   std::vector<OrderedHitTriplet>::const_iterator>;

    SeedingLayerSetsHits::LayerIndex innerLayerIndex() const { return std::get<0>(layerPairAndLayers_->layerPair()); }
    SeedingLayerSetsHits::LayerIndex middleLayerIndex() const { return std::get<1>(layerPairAndLayers_->layerPair()); }
    SeedingLayerSetsHits::LayerIndex outerLayerIndex() const { return thirdLayer_->layerIndex(); }

    std::vector<OrderedHitTriplet>::const_iterator tripletsBegin() const { return hitSets_->tripletsBegin() + thirdLayer_->tripletsBegin(); }
    std::vector<OrderedHitTriplet>::const_iterator tripletsEnd() const { return hitSets_->tripletsBegin() + thirdLayer_->tripletsEnd(); }

    const LayerHitMapCache& cache() const { return layerPairAndLayers_->cache(); }
  private:
    const IntermediateHitTriplets *hitSets_;
    const LayerPairAndLayers *layerPairAndLayers_;
    const ThirdLayer *thirdLayer_;
  };

  ////////////////////

  class RegionLayerHits {
  public:
    using LayerPairAndLayersConstIterator = std::vector<LayerPairAndLayers>::const_iterator;
    using ThirdLayerConstIterator = std::vector<ThirdLayer>::const_iterator;
    using TripletConstIterator = std::vector<OrderedHitTriplet>::const_iterator;

    class const_iterator {
    public:
      using internal_iterator_type = LayerPairAndLayersConstIterator;
      using value_type = LayerTripletHits;
      using difference_type = internal_iterator_type::difference_type;

      struct end_tag {};

      const_iterator(const IntermediateHitTriplets *hitSets, const RegionLayerHits *regionLayerHits):
        hitSets_(hitSets),
        regionLayerHits_(regionLayerHits),
        iterPair_(regionLayerHits->layerSetsBegin()),
        indThird_(iterPair_->thirdLayersBegin())
      {
        assert(regionLayerHits->layerSetsBegin() != regionLayerHits->layerSetsEnd());
      }

      const_iterator(const IntermediateHitTriplets *hitSets, const RegionLayerHits *regionLayerHits, end_tag):
        iterPair_(regionLayerHits->layerSetsEnd()),
        indThird_(std::numeric_limits<unsigned int>::max())
      {}

      value_type operator*() const {
        assert(static_cast<unsigned>(indThird_) < std::distance(hitSets_->thirdLayersBegin(), hitSets_->thirdLayersEnd()));
        return value_type(hitSets_, &(*iterPair_), &(*(hitSets_->thirdLayersBegin() + indThird_)));
      }

      const_iterator& operator++() {
        auto nextThird = indThird_+1;
        if(nextThird == iterPair_->thirdLayersEnd()) {
          ++iterPair_;
          if(iterPair_ != regionLayerHits_->layerSetsEnd()) {
            indThird_ = iterPair_->thirdLayersBegin();
          }
          else {
            indThird_ = std::numeric_limits<unsigned int>::max();
          }
        }
        else {
          indThird_ = nextThird;
        }
        return *this;
      }

      const_iterator operator++(int) {
        const_iterator clone(*this);
        operator++();
        return clone;
      }

      bool operator==(const const_iterator& other) const { return iterPair_ == other.iterPair_ && indThird_ == other.indThird_; }
      bool operator!=(const const_iterator& other) const { return !operator==(other); }

    private:
      const IntermediateHitTriplets *hitSets_;
      const RegionLayerHits *regionLayerHits_;
      internal_iterator_type iterPair_;
      unsigned int indThird_;
    };

    RegionLayerHits(const TrackingRegion* region,
                    const IntermediateHitTriplets *hitSets,
                    LayerPairAndLayersConstIterator pairBegin,
                    LayerPairAndLayersConstIterator pairEnd):
      region_(region),
      hitSets_(hitSets),
      layerSetsBegin_(pairBegin), layerSetsEnd_(pairEnd)
    {}

    const TrackingRegion& region() const { return *region_; }
    size_t layerPairAndLayersSize() const { return std::distance(layerSetsBegin_, layerSetsEnd_); }

    const_iterator begin() const {
      if(layerSetsBegin_ != layerSetsEnd_)
        return const_iterator(hitSets_, this);
      else
        return end();
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return const_iterator(hitSets_, this, const_iterator::end_tag()); }
    const_iterator cend() const { return end(); }

    // used internally
    LayerPairAndLayersConstIterator layerSetsBegin() const { return layerSetsBegin_; }
    LayerPairAndLayersConstIterator layerSetsEnd() const { return layerSetsEnd_; }

  private:
    const TrackingRegion *region_ = nullptr;
    const IntermediateHitTriplets *hitSets_ = nullptr;
    const LayerPairAndLayersConstIterator layerSetsBegin_;
    const LayerPairAndLayersConstIterator layerSetsEnd_;
  };

  ////////////////////

  using const_iterator = ihd::const_iterator<RegionLayerHits, IntermediateHitTriplets>;

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

  LayerHitMapCache *beginPair(const LayerPair& layerPair, LayerHitMapCache&& cache) {
    layerPairAndLayers_.emplace_back(layerPair, thirdLayers_.size(), std::move(cache));
    return &(layerPairAndLayers_.back().cache());
  };

  void addTriplets(const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                   const OrderedHitTriplets& triplets,
                   const std::vector<int>& thirdLayerIndex,
                   const std::vector<size_t>& permutations) {
    assert(triplets.size() == thirdLayerIndex.size());
    assert(triplets.size() == permutations.size());

    if(triplets.empty()) {
      // In absence of triplets for a layer pair simplest is just
      // remove the pair
      popPair();
      return;
    }

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
        layerPairAndLayers_.back().setThirdLayersEnd(thirdLayers_.size());
      }

      hitTriplets_.emplace_back(triplets[realIndex]);
      thirdLayers_.back().setHitsEnd(hitTriplets_.size());
    }

    regions_.back().setLayerSetsEnd(layerPairAndLayers_.size());
  }

  const SeedingLayerSetsHits& seedingLayerHits() const { return *seedingLayers_; }
  size_t regionSize() const { return regions_.size(); }
  size_t tripletsSize() const { return hitTriplets_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }

  // used internally
  std::vector<RegionIndex>::const_iterator regionsBegin() const { return regions_.begin(); }
  std::vector<RegionIndex>::const_iterator regionsEnd() const { return regions_.end(); }
  std::vector<LayerPairAndLayers>::const_iterator layerSetsBegin() const { return layerPairAndLayers_.begin(); }
  std::vector<LayerPairAndLayers>::const_iterator layerSetsEnd() const { return layerPairAndLayers_.end(); }
  std::vector<ThirdLayer>::const_iterator thirdLayersBegin() const { return thirdLayers_.begin(); }
  std::vector<ThirdLayer>::const_iterator thirdLayersEnd() const { return thirdLayers_.end(); }
  std::vector<OrderedHitTriplet>::const_iterator tripletsBegin() const { return hitTriplets_.begin(); }
  std::vector<OrderedHitTriplet>::const_iterator tripletsEnd() const { return hitTriplets_.end(); }

private:
  // to be called if no triplets are added
  void popPair() {
    layerPairAndLayers_.pop_back();
  }

  const SeedingLayerSetsHits *seedingLayers_;

  std::vector<RegionIndex> regions_;
  std::vector<LayerPairAndLayers> layerPairAndLayers_;
  std::vector<ThirdLayer> thirdLayers_;
  std::vector<OrderedHitTriplet> hitTriplets_;
};

#endif
