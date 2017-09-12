#ifndef PixelTripletNoTipGenerator_H
#define PixelTripletNoTipGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "CombinedHitTripletGenerator.h"

namespace edm { class Event; class EventSetup; } 

#include <utility>
#include <vector>


class PixelTripletNoTipGenerator : public HitTripletGeneratorFromPairAndLayers {
typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;
public:
  PixelTripletNoTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletNoTipGenerator() override;

  void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
                            const edm::Event & ev, const edm::EventSetup& es,
                            const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                            const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;
  void hitTriplets(
		   const TrackingRegion& region, 
		   OrderedHitTriplets & result,
		   const edm::EventSetup & es,
		   const HitDoublets & doublets,
		   const RecHitsSortedInPhi ** thirdHitMap,
		   const std::vector<const DetLayer *> & thirdLayerDetLayer,
		   const int nThirdLayers)override;

private:
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraHitPhiToleranceForPreFiltering;
  double theNSigma;
  double theChi2Cut;
};
#endif
