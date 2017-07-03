#ifndef CombinedHitQuadrupletGenerator_H
#define CombinedHitQuadrupletGenerator_H

/** A HitQuadrupletGenerator consisting of a set of
 *  quadruplet generators of type HitQuadrupletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerQuadruplets
 */

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"


class TrackingRegion;
class HitQuadrupletGeneratorFromTripletAndLayers;
class SeedingLayerSetsHits;

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedHitQuadrupletGenerator : public HitQuadrupletGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedHitQuadrupletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~CombinedHitQuadrupletGenerator() override;

  /// from base class
  void hitQuadruplets( const TrackingRegion& reg, OrderedHitSeeds & triplets,
      const edm::Event & ev,  const edm::EventSetup& es) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  LayerCacheType            theLayerCache;

  std::unique_ptr<HitQuadrupletGeneratorFromTripletAndLayers> theGenerator;
};
#endif
