#ifndef _PixelTripletLowPtGenerator_h_
#define _PixelTripletLowPtGenerator_h_

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CombinedHitTripletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

class TrackerGeometry;

#include <vector>

class   PixelTripletLowPtGenerator :
 public HitTripletGeneratorFromPairAndLayers {

 typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;

 public:
   PixelTripletLowPtGenerator ( const edm::ParameterSet& cfg) 
    : theConfig(cfg), thePairGenerator(0), theLayerCache(0) { theTracker = 0; }

   virtual ~PixelTripletLowPtGenerator() { delete thePairGenerator; }

   virtual void init( const HitPairGenerator & pairs,
      const std::vector<ctfseeding::SeedingLayer> & layers, LayerCacheType* layerCache);

   virtual void hitTriplets(const TrackingRegion& region, OrderedHitTriplets & trs,  const edm::Event & ev, const edm::EventSetup& es);

   const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
   const std::vector<ctfseeding::SeedingLayer> & thirdLayers() const { return theLayers; }

 private:
   void getTracker (const edm::EventSetup& es);
   GlobalPoint getGlobalPosition(const TrackingRecHit* recHit);

   const TrackerGeometry* theTracker;

   edm::ParameterSet         theConfig;
   HitPairGenerator * thePairGenerator;
   std::vector<ctfseeding::SeedingLayer> theLayers;
   LayerCacheType * theLayerCache;

  bool useClusterShape;
};

#endif
