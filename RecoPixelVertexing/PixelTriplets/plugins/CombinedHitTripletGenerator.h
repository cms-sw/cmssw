#ifndef CombinedHitTripletGenerator_H
#define CombinedHitTripletGenerator_H

/** A HitTripletGenerator consisting of a set of 
 *  triplet generators of type HitTripletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include <vector>
#include <memory>
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <string>
#include <memory>

class TrackingRegion;
class HitTripletGeneratorFromPairAndLayers;
class SeedingLayerSetsHits;

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedHitTripletGenerator : public HitTripletGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedHitTripletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~CombinedHitTripletGenerator();

  /// from base class
  virtual void hitTriplets( const TrackingRegion& reg, OrderedHitTriplets & triplets,
      const edm::Event & ev,  const edm::EventSetup& es);

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  LayerCacheType            theLayerCache;

  std::unique_ptr<HitTripletGeneratorFromPairAndLayers> theGenerator;
};
#endif
