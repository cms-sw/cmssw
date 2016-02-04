#ifndef CombinedHitTripletGenerator_H
#define CombinedHitTripletGenerator_H

/** A HitTripletGenerator consisting of a set of 
 *  triplet generators of type HitTripletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class HitTripletGeneratorFromPairAndLayers;
namespace ctfseeding { class SeedingLayer;}

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedHitTripletGenerator : public HitTripletGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedHitTripletGenerator( const edm::ParameterSet& cfg);

  virtual ~CombinedHitTripletGenerator();

  /// from base class
  virtual void hitTriplets( const TrackingRegion& reg, OrderedHitTriplets & triplets,
      const edm::Event & ev,  const edm::EventSetup& es);

private:
  void init(const edm::ParameterSet & cfg, const edm::EventSetup& es);

  mutable bool initialised;

  edm::ParameterSet         theConfig;
  LayerCacheType            theLayerCache;

  typedef std::vector<HitTripletGeneratorFromPairAndLayers* > GeneratorContainer;
  GeneratorContainer        theGenerators;
};
#endif
