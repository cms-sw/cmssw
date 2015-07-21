#ifndef PixelTripletLargeTipGenerator_H
#define PixelTripletLargeTipGenerator_H

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


class PixelTripletLargeTipGenerator : public HitTripletGeneratorFromPairAndLayers {

typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelTripletLargeTipGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~PixelTripletLargeTipGenerator();

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
                            const edm::Event & ev, const edm::EventSetup& es,
                            SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                            const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;

private:

  bool checkPhiInRange(float phi, float phi1, float phi2) const;
  std::pair<float,float> mergePhiRanges(
      const std::pair<float,float> &r1, const std::pair<float,float> &r2) const;


private:
  bool useFixedPreFiltering;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  bool useMScat;
  bool useBend;
  float dphi;
};
#endif


