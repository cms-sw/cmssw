#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGeneratorFromLayerTriplet.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCachedHit.h"
CosmicHitTripletGeneratorFromLayerTriplet::CosmicHitTripletGeneratorFromLayerTriplet
(const LayerWithHits* inner, 
 const LayerWithHits* middle, 
 const LayerWithHits* outer,
 const edm::EventSetup& iSetup)
  : TTRHbuilder(0),trackerGeometry(0),
    //theLayerCache(*layerCache), 
    theOuterLayer(outer),theMiddleLayer(middle), theInnerLayer(inner)
{

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeometry = tracker.product();
}
void CosmicHitTripletGeneratorFromLayerTriplet::hitTriplets(
  const TrackingRegion & region, OrderedHitTriplets & result,
  const edm::EventSetup& iSetup)
{
  

  if (theInnerLayer->recHits().empty()) return;
  if (theMiddleLayer->recHits().empty()) return;
  if (theOuterLayer->recHits().empty()) return;
  float radius1 =dynamic_cast<const BarrelDetLayer*>(theInnerLayer->layer())->specificSurface().radius();
  float radius2 =dynamic_cast<const BarrelDetLayer*>(theMiddleLayer->layer())->specificSurface().radius();
  float radius3 =dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer())->specificSurface().radius();
  bool seedfromoverlaps=((abs(radius1-radius2)<0.1)|| (abs(radius3-radius2)<0.1))? true : false;
  std::vector<const TrackingRecHit*>::const_iterator ohh;
  std::vector<const TrackingRecHit*>::const_iterator mhh;
  std::vector<const TrackingRecHit*>::const_iterator ihh;

  if(!seedfromoverlaps){
    for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
      const TkHitPairsCachedHit * oh=new TkHitPairsCachedHit(*ohh,iSetup);
      for(mhh=theMiddleLayer->recHits().begin();mhh!=theMiddleLayer->recHits().end();mhh++){
	const TkHitPairsCachedHit * mh=new TkHitPairsCachedHit(*mhh,iSetup);
	float z_diff =mh->z()-oh->z();
	float midy=mh->r()*sin(mh->phi());
	float outy=oh->r()*sin(oh->phi());
	float midx=mh->r()*cos(mh->phi());
	float outx=oh->r()*cos(oh->phi());
	float dxdy=abs((outx-midx)/(outy-midy));
	if((abs(z_diff)<30) && (outy*midy>0) &&(dxdy<2))	  
	  {
	    for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
	      const TkHitPairsCachedHit * ih=new TkHitPairsCachedHit(*ihh,iSetup);
	      float z_diff =mh->z()-ih->z();
	      float inny=ih->r()*sin(ih->phi());
	      float innx=ih->r()*cos(ih->phi());
	      float dxdy=abs((innx-midx)/(inny-midy));
	      if ((abs(z_diff)<30) && (inny*midy>0) &&(dxdy<2)&&(!seedfromoverlaps))
		{
		  result.push_back( OrderedHitTriplet(ih->RecHit(),mh->RecHit(), oh->RecHit()));
		}
	      delete ih;
	    } 
	  }
	delete  mh;
      }
      delete oh;
    }
  } else {
    for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
      const TkHitPairsCachedHit * oh=new TkHitPairsCachedHit(*ohh,iSetup);
      for(mhh=theMiddleLayer->recHits().begin();mhh!=theMiddleLayer->recHits().end();mhh++){
	const TkHitPairsCachedHit * mh=new TkHitPairsCachedHit(*mhh,iSetup);
 	float z_diff =mh->z()-oh->z();
	float midy=mh->r()*sin(mh->phi());
	float outy=oh->r()*sin(oh->phi());
	float midx=mh->r()*cos(mh->phi());
	float outx=oh->r()*cos(oh->phi());
	float dxdy=abs((outx-midx)/(outy-midy));
	float DeltaR=oh->r()-mh->r();
	if((abs(z_diff)<18) && (abs(oh->phi()-mh->phi())<0.05) &&(DeltaR<0)&&(dxdy<2)){
	  for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
	    const TkHitPairsCachedHit * ih=new TkHitPairsCachedHit(*ihh,iSetup);
	    float z_diff =mh->z()-ih->z();
	    float inny=ih->r()*sin(ih->phi());
	    float innx=ih->r()*cos(ih->phi());
	    float dxdy=abs((innx-midx)/(inny-midy));
	    if ((abs(z_diff)<30) && (inny*midy>0) &&(dxdy<2))
	      {
		result.push_back( OrderedHitTriplet(ih->RecHit(),mh->RecHit(), oh->RecHit()));
	      }
	    delete ih;
	  }
	}
	delete mh;
      }
      delete oh;
    }
  }
}

