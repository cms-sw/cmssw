#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGeneratorFromLayerTriplet.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <cmath>

typedef TransientTrackingRecHit::ConstRecHitPointer TkHitPairsCachedHit;

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
  bool seedfromoverlaps=((std::abs(radius1-radius2)<0.1)|| (std::abs(radius3-radius2)<0.1))? true : false;
  std::vector<const TrackingRecHit*>::const_iterator ohh;
  std::vector<const TrackingRecHit*>::const_iterator mhh;
  std::vector<const TrackingRecHit*>::const_iterator ihh;

  std::string builderName = "WithTrackAngle";
  edm::ESHandle<TransientTrackingRecHitBuilder> builder;
  iSetup.get<TransientRecHitRecord>().get(builderName, builder);

  if(!seedfromoverlaps){
    for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
       auto oh= (BaseTrackerRecHit const *)(&*ohh);
      for(mhh=theMiddleLayer->recHits().begin();mhh!=theMiddleLayer->recHits().end();mhh++){
	auto mh=  (BaseTrackerRecHit const *)(&*mhh);
	float z_diff =mh->globalPosition().z()-oh->globalPosition().z();
	float midy=mh->globalPosition().y();
	float outy=oh->globalPosition().y();
	float midx=mh->globalPosition().x();
	float outx=oh->globalPosition().x();
	float dxdy=std::abs((outx-midx)/(outy-midy));
	if((std::abs(z_diff)<30) && (outy*midy>0) &&(dxdy<2))	  
	  {
	    for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
	      auto ih= (BaseTrackerRecHit const *)(&*ihh);
	      float z_diff =mh->globalPosition().z()-ih->globalPosition().z();
	      float inny=ih->globalPosition().y();
	      float innx=ih->globalPosition().x();
	      float dxdy=std::abs((innx-midx)/(inny-midy));
	      if ((std::abs(z_diff)<30) && (inny*midy>0) &&(dxdy<2)&&(!seedfromoverlaps))
		{
		  result.push_back( OrderedHitTriplet(ih,mh,oh) );
		}
	    } 
	  }
      }
    }
  } else {
    for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
      auto oh= (BaseTrackerRecHit const *)(&*ohh);
      for(mhh=theMiddleLayer->recHits().begin();mhh!=theMiddleLayer->recHits().end();mhh++){
	auto mh= (BaseTrackerRecHit const *)(&*mhh);
 	float z_diff =mh->globalPosition().z()-oh->globalPosition().z();
	float midy=mh->globalPosition().y();
	float outy=oh->globalPosition().y();
	float midx=mh->globalPosition().x();
	float outx=oh->globalPosition().x();
	float dxdy=std::abs((outx-midx)/(outy-midy));
	float DeltaR=oh->globalPosition().perp()-mh->globalPosition().perp();
	if((std::abs(z_diff)<18) && (std::abs(oh->globalPosition().phi()-mh->globalPosition().phi())<0.05) &&(DeltaR<0)&&(dxdy<2)){
	  for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
	    auto ih= (BaseTrackerRecHit const *)(&*ihh);
	    float z_diff =mh->globalPosition().z()-ih->globalPosition().z();
	    float inny=ih->globalPosition().y();
	    float innx=ih->globalPosition().x();
	    float dxdy=std::abs((innx-midx)/(inny-midy));
	    if ((std::abs(z_diff)<30) && (inny*midy>0) &&(dxdy<2))
	      {
		result.push_back( OrderedHitTriplet(ih,mh,oh));
	      }
	  }
	}
      }
    }
  }
}

