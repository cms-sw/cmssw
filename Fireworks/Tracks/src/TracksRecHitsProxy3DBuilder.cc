// -*- C++ -*-
// $Id: TracksRecHitsProxy3DBuilder.cc,v 1.0 2008/02/22 10:37:00 Tom Danielson
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
// include file for the RecHits
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
// include file for the TrajectorySeeds
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "TEveGeoNode.h"
#include "Fireworks/Tracks/interface/TracksRecHitsProxy3DBuilder.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TEvePolygonSetProjected.h"
// For the moment, we keep the option of having points or lines to represent recHits.
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
  
void TracksRecHitsProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  TEveElementList* tList = *product;
  if ( !tList && *product ) {
    std::cout << "incorrect type" << std::endl;
    return;
  }

  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str());
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
    gEve->AddElement(tList);
  } else {
    tList->DestroyElements();
  }

  const reco::TrackCollection* tracks=0;
  iItem->get(tracks);

  if(0 == tracks ) return;

  int index=0;
  for(reco::TrackCollection::const_iterator it = tracks->begin();
      it != tracks->end();++it,++index) {
     TEveElementList* trkList = new TEveElementList(Form("track%d",index));
     gEve->AddElement(trkList,tList);
     addHits(*it, iItem, trkList);
  }

}

void
TracksRecHitsProxy3DBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
TracksRecHitsProxy3DBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   if(0!=iElements && item() && item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

void
TracksRecHitsProxy3DBuilder::addHits(const reco::Track& track,
				     const FWEventItem* iItem,
				     TEveElement* trkList)
{
   try {
      for(trackingRecHit_iterator recIt = track.recHitsBegin(); recIt != track.recHitsEnd(); ++recIt){
	 if((*recIt)->isValid()){
	    DetId detid = (*recIt)->geographicalId();
	    TString name("");
	    switch (detid.subdetId()) 
	      {
	       case SiStripDetId::TIB:
		 name = "TIB ";
		 break;
	       case SiStripDetId::TOB:
		 name = "TOB ";
		 break;
	       case SiStripDetId::TID:
		 name = "TID ";
		 break;
	       case SiStripDetId::TEC:
		 name = "TEC ";
		 break;
	       case PixelSubdetector::PixelBarrel:
		 name = "Pixel Barrel ";
		 break;
	       case PixelSubdetector::PixelEndcap:
		 name = "Pixel Endcap ";
	      }
	      
		
	      if (iItem->getGeom()) {
		 TEveGeoShape* shape = iItem->getGeom()->getShape( detid );
		 if(0!=shape) {
		    shape->SetMainTransparency(50);
		    shape->SetMainColor(iItem->defaultDisplayProperties().color());
		    shape->SetPickable(kTRUE);
		    shape->SetTitle(name + ULong_t(detid.rawId()));
		    trkList->AddElement(shape);
		 } else {
		    std::cout << "Failed to get shape extract for tracking rec hit: " << detid.rawId() << std::endl;
		 }
	      }
	   }// if the hit isValid().
	}// For Loop Over Rec hits (recIt)
     }
     catch (...) {
	std::cout << "Sorry, don't have the recHits for this event." << std::endl;
     }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(TracksRecHitsProxy3DBuilder,reco::TrackCollection,"TrackHits");

