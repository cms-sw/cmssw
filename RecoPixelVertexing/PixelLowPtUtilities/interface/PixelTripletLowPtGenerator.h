#ifndef _PixelTripletLowPtGenerator_h_
#define _PixelTripletLowPtGenerator_h_

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletFilter.h"

class TrackerGeometry;
class TripletFilter;

#include <vector>

class   PixelTripletLowPtGenerator :
 public HitTripletGeneratorFromPairAndLayers {


 public:
   PixelTripletLowPtGenerator( const edm::ParameterSet& cfg) 
     : theTracker(0), theFilter(0), ps(cfg), thePairGenerator(0), theLayerCache(0)
   {  }

   virtual ~PixelTripletLowPtGenerator() { delete thePairGenerator; delete theFilter; }

   virtual void init( const HitPairGenerator & pairs,
      const std::vector<ctfseeding::SeedingLayer> & layers, LayerCacheType* layerCache);

   virtual void hitTriplets(const TrackingRegion& region, OrderedHitTriplets & trs,  const edm::Event & ev, const edm::EventSetup& es);

   const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
   const std::vector<ctfseeding::SeedingLayer> & thirdLayers() const { return theLayers; }

 private:
  void getTracker (const edm::EventSetup& es);
  GlobalPoint getGlobalPosition(const TrackingRecHit* recHit);

  const TrackerGeometry * theTracker;
  TripletFilter * theFilter;

  edm::ParameterSet         ps;
  HitPairGenerator * thePairGenerator;
  std::vector<ctfseeding::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;

  double nSigMultipleScattering;
  double rzTolerance;
  double maxAngleRatio;

  std::string builderName;
  bool checkMultipleScattering;
  bool checkClusterShape;
 
};

#endif
