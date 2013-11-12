#ifndef PixelTripletHLTGenerator_H
#define PixelTripletHLTGenerator_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "CombinedHitTripletGenerator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

#include <utility>
#include <vector>

class SeedComparitor;

class PixelTripletHLTGenerator : public HitTripletGeneratorFromPairAndLayers {

typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelTripletHLTGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~PixelTripletHLTGenerator();

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                        std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) override;

  void init( const HitPairGenerator & pairs, LayerCacheType* layerCache) override;

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs, 
      const edm::Event & ev, const edm::EventSetup& es);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }

private:
  bool checkPhiInRange(float phi, float phi1, float phi2) const;
  std::pair<float,float> mergePhiRanges(
      const std::pair<float,float> &r1, const std::pair<float,float> &r2) const; 

private:
  HitPairGenerator * thePairGenerator;
  std::vector<SeedingLayerSetsHits::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;

  bool useFixedPreFiltering;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  bool useMScat;
  bool useBend;
  float dphi;
  std::unique_ptr<SeedComparitor> theComparitor;

};
#endif


