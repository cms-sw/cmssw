// -*- C++ -*-
// $Id: TracksRecHitsUtil.cc,v 1.2 2009/01/16 10:37:00 amraktad
//

#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"

#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveElement.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Fireworks/Core/interface/FWDetIdInfo.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

void TracksRecHitsUtil::buildTracksRecHits(const FWEventItem* iItem, TEveElementList** product)
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
       it != tracks->end(); ++it,++index) {
      TEveElementList* trkList = new TEveElementList(Form("track%d",index));
      gEve->AddElement(trkList,tList);
      addHits(*it, iItem, trkList);
   }

}

void
TracksRecHitsUtil::addHits(const reco::Track& track,
                           const FWEventItem* iItem,
                           TEveElement* trkList)
{
   try {
     std::set<unsigned int> ids;
      for(trackingRecHit_iterator recIt = track.recHitsBegin(); recIt != track.recHitsEnd(); ++recIt){
         if((*recIt)->isValid()){
            DetId detid = (*recIt)->geographicalId();
            TString name("");
	    switch (detid.det())
	      {
	      case DetId::Tracker:
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
                  default:
                    break;
		  }
		break;

	      case DetId::Muon:
		switch (detid.subdetId())
		  {
		  case MuonSubdetId::DT:
		    name = "DT";
		    detid = DetId(DTChamberId(detid)); // get rid of layer bits
		    break;
		  case MuonSubdetId::CSC:
		    name = "CSC";
		    break;
		  case MuonSubdetId::RPC:
		    name = "RPC";
		    break;
                  default:
                    break;
		  }
		break;

              default:
                break;
	      }
	    if ( ! ids.insert(detid.rawId()).second ) continue;
            if (iItem->getGeom()) {
               TEveGeoShape* shape = iItem->getGeom()->getShape( detid );
               if(0!=shape) {
                  shape->SetMainTransparency(65);
                  shape->SetPickable(kTRUE);
                  shape->SetMainColor(iItem->defaultDisplayProperties().color());
                  shape->SetTitle(name + ULong_t(detid.rawId()));
                  trkList->AddElement(shape);
               } else {
		 std::cout << "Failed to get shape extract for a tracking rec hit: " << 
		   "\n" << FWDetIdInfo::info(detid) << std::endl;
               }
            }
         }  // if the hit isValid().
      }  // For Loop Over Rec hits (recIt)
   }
   catch (...) {
      std::cout << "Sorry, don't have the recHits for this event." << std::endl;
   }
}
