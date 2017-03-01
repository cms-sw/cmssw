#ifndef CosmicLayerTriplets_H
#define CosmicLayerTriplets_H

/** \class CosmicLayerTriplets
 * find all (resonable) pairs of pixel layers
 */
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include <vector>
class CosmicLayerTriplets {
public:
  CosmicLayerTriplets() {};
  ~CosmicLayerTriplets();
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);
 typedef std::pair<SeedLayerPairs::LayerPair, std::vector<const LayerWithHits*> > LayerPairAndLayers;


  //  virtual std::vector<LayerPair> operator()() const;
  //  std::vector<LayerTriplet> operator()() ;
  std::vector<LayerPairAndLayers> layers();

private:

  //definition of the map 
 

  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;
  LayerWithHits *lh4;

  edm::ESWatcher<TrackerRecoGeometryRecord> watchTrackerGeometry_;

   std::vector<BarrelDetLayer const*> bl;
   //MP
   std::vector<LayerWithHits*> allLayersWithHits;
 public:
 
   void init(const SiStripRecHit2DCollection &collstereo,
	     const SiStripRecHit2DCollection &collrphi,
	     const SiStripMatchedRecHit2DCollection &collmatched,
	     std::string geometry,
	     const edm::EventSetup& iSetup);

 private:
 std::string _geometry;
};




#endif
