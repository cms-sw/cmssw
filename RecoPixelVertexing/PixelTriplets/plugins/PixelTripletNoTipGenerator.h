#ifndef PixelTripletNoTipGenerator_H
#define PixelTripletNoTipGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "CombinedHitTripletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

namespace edm { class Event; class EventSetup; } 

#include <utility>
#include <vector>


class PixelTripletNoTipGenerator : public HitTripletGeneratorFromPairAndLayers {
typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;
public:
  PixelTripletNoTipGenerator(const edm::ParameterSet& cfg);

  virtual ~PixelTripletNoTipGenerator() { delete thePairGenerator; }

  virtual void init( const HitPairGenerator & pairs,
      const std::vector<ctfseeding::SeedingLayer> & layers, LayerCacheType* layerCache);

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
      const edm::Event & ev, const edm::EventSetup& es);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
  const std::vector<ctfseeding::SeedingLayer> & thirdLayers() const { return theLayers; }

private:
  HitPairGenerator * thePairGenerator;
  std::vector<ctfseeding::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraHitPhiToleranceForPreFiltering;
  double theNSigma;
  double theChi2Cut;
};
#endif
