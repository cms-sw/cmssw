#ifndef CosmicHitTripletGeneratorFromLayerTriplet_h
#define CosmicHitTripletGeneratorFromLayerTriplet_h

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
class DetLayer;
class TrackingRegion;
class LayerWithHits;

class CosmicHitTripletGeneratorFromLayerTriplet : public HitTripletGeneratorFromPairAndLayers {

public:


  CosmicHitTripletGeneratorFromLayerTriplet( 
				const LayerWithHits* inner, 
				const LayerWithHits* middle, 
				const LayerWithHits* outer, 
				const edm::EventSetup& iSetup);
  virtual ~CosmicHitTripletGeneratorFromLayerTriplet() { }

  virtual OrderedHitTriplets hitTriplets( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
    return HitTripletGenerator::hitTriplets(region, iSetup);
  }
  virtual void hitTriplets( const TrackingRegion& ar, OrderedHitTriplets & ap,const edm::EventSetup& iSetup);

  virtual CosmicHitTripletGeneratorFromLayerTriplet* clone() const {
    return new CosmicHitTripletGeneratorFromLayerTriplet(*this);
  }
  void init( const HitPairGenerator & pairs,
	     std::vector<const LayerWithHits*> layers, LayerCacheType* layerCache){}
  const LayerWithHits* innerLayer() const { return theInnerLayer; }
  const LayerWithHits* middleLayer() const { return theMiddleLayer; }
  const LayerWithHits* outerLayer() const { return theOuterLayer; }

private:
  const TransientTrackingRecHitBuilder * TTRHbuilder;
  const TrackerGeometry* trackerGeometry;
  const LayerWithHits* theOuterLayer;  
  const LayerWithHits* theMiddleLayer; 
  const LayerWithHits* theInnerLayer; 
  const DetLayer* innerlay;
  const DetLayer* outerlay;
  const DetLayer* middlelay;

};

#endif
