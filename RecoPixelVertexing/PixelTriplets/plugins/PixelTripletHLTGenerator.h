#ifndef PixelTripletHLTGenerator_H
#define PixelTripletHLTGenerator_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

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
  PixelTripletHLTGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC): PixelTripletHLTGenerator(cfg, iC) {}
  PixelTripletHLTGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletHLTGenerator() override;

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char *fillDescriptionsLabel() { return "pixelTripletHLT"; }

  void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
                            const edm::Event & ev, const edm::EventSetup& es,
                            const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                            const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;

  void hitTriplets(const TrackingRegion& region, OrderedHitTriplets& trs,
                   const edm::Event& ev, const edm::EventSetup& es,
                   const HitDoublets& doublets,
                   const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                   std::vector<int> *tripletLastLayerIndex,
                   LayerCacheType& layerCache);

    void hitTriplets(
	const TrackingRegion& region, 
	OrderedHitTriplets & result,
	const edm::EventSetup & es,
	const HitDoublets & doublets,
	const RecHitsSortedInPhi ** thirdHitMap,
	const std::vector<const DetLayer *> & thirdLayerDetLayer,
	const int nThirdLayers)override;

  void hitTriplets(const TrackingRegion& region, OrderedHitTriplets & result,
                   const edm::EventSetup & es,
                   const HitDoublets & doublets,
                   const RecHitsSortedInPhi ** thirdHitMap,
                   const std::vector<const DetLayer *> & thirdLayerDetLayer,
                   const int nThirdLayers,
                   std::vector<int> *tripletLastLayerIndex);

private:
  const bool useFixedPreFiltering;
  const float extraHitRZtolerance;
  const float extraHitRPhitolerance;
  const bool useMScat;
  const bool useBend;
  const float dphi;
  std::unique_ptr<SeedComparitor> theComparitor;

};
#endif


