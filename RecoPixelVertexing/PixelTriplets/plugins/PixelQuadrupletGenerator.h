#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "CombinedHitQuadrupletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"

#include <utility>
#include <vector>

class SeedComparitor;

class PixelQuadrupletGenerator : public HitQuadrupletGeneratorFromTripletAndLayers {

typedef CombinedHitQuadrupletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelQuadrupletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~PixelQuadrupletGenerator();

  virtual void hitQuadruplets( const TrackingRegion& region, OrderedHitSeeds& result,
                               const edm::Event & ev, const edm::EventSetup& es,
                               SeedingLayerSetsHits::SeedingLayerSet tripletLayers,
                               const std::vector<SeedingLayerSetsHits::SeedingLayer>& fourthLayers) override;

private:
  std::unique_ptr<SeedComparitor> theComparitor;

  const double extraHitRZtolerance;
  const double extraHitRPhitolerance;
  const double maxChi2;
  const bool keepTriplets;
};



