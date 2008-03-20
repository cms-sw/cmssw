#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/TrackDetailView.h"

TrackDetailView::TrackDetailView () 
{

}

TrackDetailView::~TrackDetailView ()
{

}

void TrackDetailView::build (TEveElementList **product, const FWModelId &id)
{
     m_item = id.item();
     // printf("calling ElectronDetailView::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(m_item->defaultDisplayProperties().color());
     } else {
	  return;
// 	  tList->DestroyElements();
     }

     TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
     //units are Tesla
     rnrStyle->SetMagField( -4.0);
     //get this from geometry, units are CM
     rnrStyle->SetMaxR(120.0);
     rnrStyle->SetMaxZ(300.0);    
     
     const reco::TrackCollection* tracks=0;
     m_item->get(tracks);
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
	  if (index != id.index())
	       continue;
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
	  trk->SetMainColor(m_item->defaultDisplayProperties().color());
	  trk->MakeTrack();
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
			 const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( detid );
			 // And here are the shapes of the hit modules
			 TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract( detid );
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

	       recHitsLines->SetMainColor(m_item->defaultDisplayProperties().color());
	       trkList->AddElement(recHitsLines);
	  }
	  catch (...) {
	       //      std::cout << "Sorry, don't have the recHits for this event." << std::endl;
	  }
    
     }
}
