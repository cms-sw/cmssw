#ifndef CosmicHitTripletGeneratorFromLayerTriplet_h
#define CosmicHitTripletGeneratorFromLayerTriplet_h

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class DetLayer;
class TrackingRegion;
class LayerWithHits;


class CosmicHitTripletGeneratorFromLayerTriplet {

public:


  CosmicHitTripletGeneratorFromLayerTriplet( 
				const LayerWithHits* inner, 
				const LayerWithHits* middle, 
				const LayerWithHits* outer, 
				const edm::EventSetup& iSetup);
  ~CosmicHitTripletGeneratorFromLayerTriplet() { }

  void hitTriplets( const TrackingRegion& ar, OrderedHitTriplets & ap, const edm::EventSetup& iSetup);

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
