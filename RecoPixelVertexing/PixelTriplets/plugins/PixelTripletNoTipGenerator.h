#ifndef PixelTripletNoTipGenerator_H
#define PixelTripletNoTipGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "CombinedHitTripletGenerator.h"

namespace edm { class Event; class EventSetup; } 

#include <utility>
#include <vector>


class PixelTripletNoTipGenerator : public HitTripletGeneratorFromPairAndLayers {
typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;
public:
  PixelTripletNoTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~PixelTripletNoTipGenerator() { delete thePairGenerator; }

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                        std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) override;

  void init( const HitPairGenerator & pairs, LayerCacheType* layerCache) override;

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
      const edm::Event & ev, const edm::EventSetup& es);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }

private:
  HitPairGenerator * thePairGenerator;
  std::vector<SeedingLayerSetsHits::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraHitPhiToleranceForPreFiltering;
  double theNSigma;
  double theChi2Cut;
};
#endif
