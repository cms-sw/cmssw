#ifndef CosmicLayerTriplets_H
#define CosmicLayerTriplets_H

/** \class CosmicLayerTriplets
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"


#include <vector>
class CosmicLayerTriplets {
public:
  CosmicLayerTriplets():isFirstCall(true){};
  ~CosmicLayerTriplets();
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);
 typedef std::pair<SeedLayerPairs::LayerPair, std::vector<const LayerWithHits*> > LayerPairAndLayers;


  //  virtual std::vector<LayerPair> operator()() const;
  //  std::vector<LayerTriplet> operator()() ;
  std::vector<LayerPairAndLayers> layers();

private:

  //definition of the map 
 

  SiStripRecHit2DCollection::range rphi_range1;
  SiStripRecHit2DCollection::range rphi_range2;
  SiStripRecHit2DCollection::range rphi_range3;
  SiStripRecHit2DCollection::range rphi_range4;

  SiStripRecHit2DCollection::range stereo_range1;
  SiStripRecHit2DCollection::range stereo_range2;
  SiStripRecHit2DCollection::range stereo_range3;

  SiStripMatchedRecHit2DCollection::range match_range1;
  SiStripMatchedRecHit2DCollection::range match_range2;
  TrackerLayerIdAccessor acc;
  
  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;
  LayerWithHits *lh4;

   std::vector<BarrelDetLayer*> bl;
   std::vector<ForwardDetLayer*> fpos;
   std::vector<ForwardDetLayer*> fneg;
   //MP
   std::vector<LayerWithHits*> allLayersWithHits;
   bool isFirstCall;
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
