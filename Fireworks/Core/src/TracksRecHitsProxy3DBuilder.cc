// -*- C++ -*-
//
// Package:     Core
// Class  :     TracksRecHitsProxy3DBuilder
// 
/**\class TracksRecHitsProxy3DBuilder TracksRecHitsProxy3DBuilder.h Fireworks/Core/interface/TracksRecHitsProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// Based on
// $Id: TracksRecHitsProxy3DBuilder.cc,v 1.3 2008/06/09 19:54:03 chrjones Exp $
// New File:
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
#include "Fireworks/Core/interface/TracksRecHitsProxy3DBuilder.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TEvePolygonSetProjected.h"
// For the moment, we keep the option of having points or lines to represent recHits.
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"


void TracksRecHitsProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  std::cout <<"build called"<<std::endl;
  
  //  Original commented out here
  //  TEveTrackList* tList = dynamic_cast<TEveTrackList*>*product;

  TEveElementList* tList = *product;
  if ( !tList && *product ) {
    std::cout << "incorrect type" << std::endl;
    return;
  }
  
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str());
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
    //Original commented out here
    //TEveTrackPropagator* rnrStyle = tList->GetPropagator();
    //units are Tesla
    //rnrStyle->SetMagField( -4.0);
    //get this from geometry, units are CM
    //rnrStyle->SetMaxR(120.0);
    //rnrStyle->SetMaxZ(300.0);    
    gEve->AddElement(tList);
  } else {
    tList->DestroyElements();
  }
  
  TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
  //units are Tesla
  rnrStyle->SetMagField( -4.0);
  //get this from geometry, units are CM
  rnrStyle->SetMaxR(120.0);
  rnrStyle->SetMaxZ(300.0);    
  
  const reco::TrackCollection* tracks=0;
  iItem->get(tracks);
  //fwlite::Handle<reco::TrackCollection> tracks;
  //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
  
  if(0 == tracks ) {
    std::cout <<"failed to get Tracks"<<std::endl;
    return;
  }

  //  Original Commented out here
  //  TEveTrackPropagator* rnrStyle = tList->GetPropagator();
  
  int index=0;
  //cout <<"----"<<endl;
  TEveRecTrack t;
  
  t.fBeta = 1.;
  for(reco::TrackCollection::const_iterator it = tracks->begin();
	it != tracks->end();++it,++index) {
    t.fP = TEveVector(it->px(),
		      it->py(),
		      it->pz());
    t.fV = TEveVector(it->vx(),
		      it->vy(),
		      it->vz());
    t.fSign = it->charge();

    TEveElementList* trkList = new TEveElementList(Form("track%d",index));
    gEve->AddElement(trkList,tList);  
    TEveTrack* trk = new TEveTrack(&t,rnrStyle);
    trk->SetMainColor(iItem->defaultDisplayProperties().color());
    trkList->AddElement(trk);

    /* The addition of RecHits is in a try-catch block to avoid exceptions
     * occuring when TrackExtras and/or RecHits aren't available.
     */

    try {
      
      //      HitPattern pattern = (*it).HitPattern();

      Int_t nHits = (*it).recHitsSize();
      // const reco::HitPattern& p = (*it).hitPattern();
      // If we have muon tracks, then this is going to be bad

      // Declare point set with size of the overall number of recHits
      TEvePointSet *recHitsMarker = new TEvePointSet(nHits);
      // Make a TEveStraightLineSet for the line implementation of recHits
      TEveStraightLineSet *recHitsLines = new TEveStraightLineSet();
      Int_t globalRecHitIndex = 0;
      Double_t localRecHitPoint[3];
      Double_t globalRecHitPoint[3];
      
      for(trackingRecHit_iterator recIt = it->recHitsBegin(); recIt != it->recHitsEnd(); ++recIt){
	if((*recIt)->isValid()){
	  DetId detid = (*recIt)->geographicalId();
	  LocalPoint lp = (*recIt)->localPosition();
	  // localError we don't use just yet.
	  //	  LocalError le = (*recIt)->localPositionError();
	  // Here's the local->global transformation matrix
	  const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( detid );
	  // And here are the shapes of the hit modules
	  TEveGeoShapeExtract* extract = iItem->getGeom()->getExtract( detid );
	  if(0!=extract) {
	    TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
	    shape->SetMainTransparency(50);
	    shape->SetMainColor(tList->GetMainColor());
	    trkList->AddElement(shape);
	  }

	  // set local position to a more digestable format
	  localRecHitPoint[0] = lp.x();
	  localRecHitPoint[1] = lp.y();
	  localRecHitPoint[2] = 0;
	  
	  // reset global position
	  globalRecHitPoint[0] = 0;
	  globalRecHitPoint[1] = 0;
	  globalRecHitPoint[2] = 0;


	  /*  Do segments for the recHits.  Right now they're going to be of arbitrary length.  
	      Eventually, we'll have detector geom in here, but for now, we'll just use 5cm 
	      segments in the local y.  These transform nicely.
	  */
	  
	  Double_t localRecHitInnerPoint[3];
	  Double_t localRecHitOuterPoint[3];
	  Double_t globalRecHitInnerPoint[3];
	  Double_t globalRecHitOuterPoint[3];
	  
	  localRecHitInnerPoint[0] = 0;
	  localRecHitInnerPoint[1] = 2.5;
	  localRecHitInnerPoint[2] = 0;
	  
	  localRecHitOuterPoint[0] = 0;
	  localRecHitOuterPoint[1] = -2.5;
	  localRecHitOuterPoint[2] = 0;
	  
	  // Transform from local to global, if possible
	  if ( matrix ) {
	    matrix->LocalToMaster(localRecHitPoint, globalRecHitPoint);
	    matrix->LocalToMaster(localRecHitInnerPoint, globalRecHitInnerPoint);
	    matrix->LocalToMaster(localRecHitOuterPoint, globalRecHitOuterPoint);
	    // Now we add the point to the TEvePointSet
	    recHitsMarker->SetPoint(globalRecHitIndex, globalRecHitPoint[0], 
				    globalRecHitPoint[1], globalRecHitPoint[2]);
	    recHitsLines->AddLine(globalRecHitInnerPoint[0], globalRecHitInnerPoint[1], globalRecHitInnerPoint[2],
				  globalRecHitOuterPoint[0], globalRecHitOuterPoint[1], globalRecHitOuterPoint[2]);
	    globalRecHitIndex++;
	  }	 

	}// if the hit isValid().
      }// For Loop Over Rec hits (recIt)

      // This saved just in case we want to use points for recHits.
      //recHitsMarker->SetMainColor(iItem->defaultDisplayProperties().color());
      //recHitsMarker->SetMarkerSize(1);
      //trkList->AddElement(recHitsMarker);            

      recHitsLines->SetMainColor(iItem->defaultDisplayProperties().color());
      trkList->AddElement(recHitsLines);
    }
    catch (...) {
      //      std::cout << "Sorry, don't have the recHits for this event." << std::endl;
    }
    
  }
  
}
REGISTER_FWRPZDATAPROXYBUILDER(TracksRecHitsProxy3DBuilder,reco::TrackCollection,"TrackHits");

