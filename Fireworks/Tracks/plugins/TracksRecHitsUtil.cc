// -*- C++ -*-
// $Id: TracksRecHitsUtil.cc,v 1.2 2009/01/16 10:37:00 amraktad
//

#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

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

void TracksRecHitsUtil::buildTracksRecHits(const FWEventItem* iItem, 
					   TEveElementList** product,
					   bool showHits, bool showModules)
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
      if (showHits)    addHits(*it, iItem, trkList, false);
      if (showModules) addModules(*it, iItem, trkList, false);
   }

}


void
TracksRecHitsUtil::addHits(const reco::Track& track,
                           const FWEventItem* iItem,
                           TEveElement* trkList,
                           bool addNearbyHits)
{
    std::vector<TVector3> pixelPoints;
    fireworks::pushPixelHits(pixelPoints, *iItem, track);
    TEveElementList* pixels = new TEveElementList("Pixels");
    trkList->AddElement(pixels);
    if (addNearbyHits) {
        // get the extra hits
        std::vector<TVector3> pixelExtraPoints;
        fireworks::pushNearbyPixelHits(pixelExtraPoints, *iItem, track);
        // draw first the others
        fireworks::addTrackerHits3D(pixelExtraPoints, pixels, kRed, 1);
        // then the good ones, so they're on top
        fireworks::addTrackerHits3D(pixelPoints, pixels, kGreen, 1);
    } else {
        // just add those points with the default color
        fireworks::addTrackerHits3D(pixelPoints, pixels, iItem->defaultDisplayProperties().color(), 1);
    }

    // strips
    TEveElementList* strips = new TEveElementList("Strips");
    trkList->AddElement(strips);
    fireworks::addSiStripClusters(iItem, track, strips, iItem->defaultDisplayProperties().color(), addNearbyHits);
    
}

void
TracksRecHitsUtil::addModules(const reco::Track& track,
                           const FWEventItem* iItem,
                           TEveElement* trkList,
                           bool addLostHits)
{
   try {
     std::set<unsigned int> ids;
      for(trackingRecHit_iterator recIt = track.recHitsBegin(); recIt != track.recHitsEnd(); ++recIt){
         DetId detid = (*recIt)->geographicalId();
         if (!addLostHits && !(*recIt)->isValid()) continue;
         if(detid.rawId() != 0){
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
                  switch ((*recIt)->type()) {
                      case TrackingRecHit::valid:
                          shape->SetMainColor(iItem->defaultDisplayProperties().color());
                          break;
                      case TrackingRecHit::missing:
                          name += "LOST ";
                          shape->SetMainColor(kRed);
                          break;
                      case TrackingRecHit::inactive:
                          name += "INACTIVE ";
                          shape->SetMainColor(28);
                          break;
                      case TrackingRecHit::bad:
                          name += "BAD ";
                          shape->SetMainColor(218);
                          break;
                  }
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
