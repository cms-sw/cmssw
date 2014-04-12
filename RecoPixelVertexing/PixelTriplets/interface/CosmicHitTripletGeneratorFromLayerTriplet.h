#ifndef CosmicHitTripletGeneratorFromLayerTriplet_h
#define CosmicHitTripletGeneratorFromLayerTriplet_h

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class DetLayer;
class TrackingRegion;
class LayerWithHits;


class CosmicHitTripletGeneratorFromLayerTriplet : public HitTripletGenerator {

public:


  CosmicHitTripletGeneratorFromLayerTriplet( 
				const LayerWithHits* inner, 
				const LayerWithHits* middle, 
				const LayerWithHits* outer, 
				const edm::EventSetup& iSetup);
  virtual ~CosmicHitTripletGeneratorFromLayerTriplet() { }

  virtual void hitTriplets( const TrackingRegion& ar, OrderedHitTriplets & ap, const edm::EventSetup& iSetup);

  virtual void hitTriplets( const TrackingRegion& ar, OrderedHitTriplets & ap, const edm::Event& ev, const edm::EventSetup& iSetup) {}

  virtual CosmicHitTripletGeneratorFromLayerTriplet* clone() const {
    return new CosmicHitTripletGeneratorFromLayerTriplet(*this);
  }
  void init( const HitPairGenerator & pairs,
	     const std::vector<const LayerWithHits*>& layers ){}
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
