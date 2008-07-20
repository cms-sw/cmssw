// -*- C++ -*-
//
// Package:     Core
// Class  :     TracksProxy3DBuilder
// 
/**\class TracksProxy3DBuilder TracksProxy3DBuilder.h Fireworks/Core/interface/TracksProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: TracksProxy3DBuilder.cc,v 1.13 2008/07/17 10:04:17 dmytro Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"
// #include <sstream>

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Fireworks/Core/interface/TracksProxy3DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"

void TracksProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;
   
   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"trackerMuons",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
    } else {
      tList->DestroyElements();
    }

    const reco::TrackCollection* tracks=0;
    iItem->get(tracks);
    
    if(0 == tracks ) return;
    
    TEveTrackPropagator* propagator = new TEveTrackPropagator();

   // if auto field estimation mode, do extra loop over the tracks.
   if ( CmsShowMain::isAutoField() )
     for(reco::TrackCollection::const_iterator it = tracks->begin(); it != tracks->end();++it) {
	if ( fabs( it->eta() ) > 2.0 || it->pt() < 1 ) continue;
	double estimate = fw::estimate_field(*it);
	if ( estimate < 0 ) continue;
	CmsShowMain::guessFieldIsOn(estimate>2.0);
     }

   // if ( CmsShowMain::isAutoField() )
   //  printf("Field auto mode status: field=%0.1f, #estimates=%d\n",
   //	    CmsShowMain::getMagneticField(), CmsShowMain::getFieldEstimates());
   propagator->SetMagField( - CmsShowMain::getMagneticField() );
   propagator->SetMaxR(123.0);
   propagator->SetMaxZ(300.0);
    int index=0;
    for(reco::TrackCollection::const_iterator it = tracks->begin();
	it != tracks->end();++it,++index) {
       const unsigned int bufSize = 1024;
       char title[bufSize];
       char name[bufSize];
       snprintf(name,  bufSize,"track%d",index);
       snprintf(title, bufSize,"Track %d, Pt: %0.1f GeV",index,it->pt());
       TEveCompound* trkList = new TEveCompound(name, title);
       trkList->OpenCompound();
       //guarantees that CloseCompound will be called no matter what happens
       boost::shared_ptr<TEveCompound> sentry(trkList,boost::mem_fn(&TEveCompound::CloseCompound));
       trkList->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
       trkList->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
       
       std::vector<TEveTrack*> list = 
	 prepareTrack( *it, propagator, trkList, iItem->defaultDisplayProperties().color() );
       for ( std::vector<TEveTrack*>::iterator trk = list.begin(); trk != list.end(); ++ trk )
	 {	    
	    (*trk)->MakeTrack();
	    trkList->AddElement( *trk );
	 }
       
       gEve->AddElement(trkList,tList);
      // printf("track pt: %.1f, eta: %0.1f => B: %0.2f T\n", it->pt(), it->eta(), fw::estimate_field(*it));
    }
}


std::vector<TEveTrack*> 
TracksProxy3DBuilder::prepareTrack(const reco::Track& track, 
				   TEveTrackPropagator* propagator,
				   TEveElement* trackList,
				   Color_t color)
{
   // We always propagate each track from its point of closest 
   // approach wrt 0,0,0. If extra information is available, we try
   // to make use of it
   // Due to restrictions in the projections are created, all track
   // fragments have to fly outward without crossing Y=0

   // cases to considere:
   // 1) track passes POCA and crosses the detector
   // 2) track moves inward
   //   a) one side
   //   b) cross the detector (excluding 1)
   // 3) normal pp collisions
   // 4) AOD only information (single state)

   const float zeroOffset = 1e-6;
   std::vector<TEveTrack*> result;
   
   if ( ! track.extra().isAvailable() )
     return prepareSimpleTrack(track,propagator,trackList,color);
   
   // weird case, crossed tracker, but effectively only 2 states
   // screw it.
   if ( track.innerPosition().y()*track.outerPosition().y() < 0	&&
	(
	 ( fabs(track.vx()-track.innerPosition().x())<0.01 &&
	   fabs(track.vy()-track.innerPosition().y())<0.01 &&
	   fabs(track.vz()-track.innerPosition().z())<0.01 ) ||
	 ( fabs(track.vx()-track.outerPosition().x())<0.01 &&
	   fabs(track.vy()-track.outerPosition().y())<0.01 &&
	   fabs(track.vz()-track.outerPosition().z())<0.01 ) ) )
     return prepareSimpleTrack(track,propagator,trackList,color);
   
   /*
     {
	// AOD case, draw what we can
      TEveRecTrack t;
      t.fBeta = 1.;
      if ( track.vy()*track.py() > 0 )
	t.fV = TEveVector(track.vx(), track.vy(), track.vz());
      else
	t.fV = TEveVector(track.vx()-track.px()/track.py()*track.vy(),
			  track.py()>0?zeroOffset:-zeroOffset,
			  track.vz()-track.pz()/track.py()*track.vy());
      t.fP = TEveVector(track.px(), track.py(), track.pz());
      t.fSign = track.charge();
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetMainColor(color);
      trk->MakeTrack();
      trackList->AddElement( trk );
      return trk;
   } // Done with AOD
    */

   TEvePathMark mark1( TEvePathMark::kDaughter );
   mark1.fV = TEveVector( track.innerPosition().x(), 
			  track.innerPosition().y(), 
			  track.innerPosition().z() );
	       
   TEvePathMark mark2( TEvePathMark::kDaughter );
   mark2.fV = TEveVector( track.outerPosition().x(), 
			  track.outerPosition().y(), 
			  track.outerPosition().z() );
   TEvePathMark markPOCA( TEvePathMark::kDaughter );
   markPOCA.fV = TEveVector( track.vx(), track.vy(), track.vz() );

   TEvePointSet* states = new TEvePointSet("states");
   states->SetMarkerStyle(2);
   states->SetMarkerSize(0.2); //cm?
   states->SetMarkerColor(color);
   states->SetNextPoint( track.innerPosition().x(),
			 track.innerPosition().y(),
			 track.innerPosition().z() );
   states->SetNextPoint( track.outerPosition().x(),
			 track.outerPosition().y(),
			 track.outerPosition().z() );
   trackList->AddElement( states );
   if ( ( track.innerPosition().x()*track.outerPosition().x() +  
	  track.innerPosition().y()*track.outerPosition().y() < 0 )
	&&
	( track.innerPosition().x()*track.px() +  
	  track.innerPosition().y()*track.py() < 0 ) )
     { 
	// std::cout << "Track " << track.pt() << " crossed the detector" << std::endl;
	// track crossed the detector, POCA is in between inner 
	// and outer states and states are properly ordered, i.e.
	// momentum is pointing from inner state toward POCA and 
	// outerstate
	// we make two track fragments
	   
	bool crossedY0 = (track.innerPosition().y()*track.outerPosition().y() < 0);
	
	if ( ! crossedY0 )  {
	   TEveRecTrack t;
	   t.fBeta = 1.;
	   t.fV = TEveVector(track.vx(), track.vy(), track.vz());
	   
	   t.fP = TEveVector(track.px(), track.py(), track.pz());
	   t.fSign = track.charge();
	   TEveTrack* trkAlong = new TEveTrack(&t,propagator);
	   trkAlong->AddPathMark( mark2 );
	   trkAlong->SetMainColor(color);
	   // trkAlong->MakeTrack();
	   // trackList->AddElement(trkAlong);
	   result.push_back(trkAlong);
	      
	   t.fP = TEveVector(-track.px(), -track.py(), -track.pz());
	   t.fSign = -track.charge();
	   TEveTrack* trkOpposite = new TEveTrack(&t,propagator);
	   trkOpposite->AddPathMark( mark1 );
	   trkOpposite->SetMainColor(color);
	   // trkOpposite->MakeTrack();
	   // trackList->AddElement(trkOpposite);
	   result.push_back(trkOpposite);
	   return result;
	} else {
	   // find intersection point with y=0 
	   // (straight line propagation along/oppisite to momentum
	   TEveRecTrack t;
	   t.fBeta = 1.;
	   t.fV = TEveVector(track.vx()-track.px()/track.py()*track.vy(),
			     track.py()>0?zeroOffset:-zeroOffset,
			     track.vz()-track.pz()/track.py()*track.vy());
	   t.fP = TEveVector(track.px(), track.py(), track.pz());
	   t.fSign = track.charge();
	   TEveTrack* trkAlong = new TEveTrack(&t,propagator);
	   trkAlong->AddPathMark( mark2 );
	   trkAlong->SetMainColor(color);
	   // trkAlong->MakeTrack();
	   // trackList->AddElement(trkAlong);
	   result.push_back(trkAlong);
	   
	   t.fV = TEveVector(track.vx()-track.px()/track.py()*track.vy(),
			     track.py()>0?-zeroOffset:zeroOffset,
			     track.vz()-track.pz()/track.py()*track.vy());
	   t.fP = TEveVector(-track.px(), -track.py(), -track.pz());
	   t.fSign = -track.charge();
	   TEveTrack* trkOpposite = new TEveTrack(&t,propagator);
	   trkOpposite->AddPathMark( mark1 );
	   trkOpposite->SetMainColor(color);
	   // trkOpposite->MakeTrack();
	   // trackList->AddElement(trkOpposite);
	   result.push_back(trkOpposite);
	   return result;
	}
     } 
      
   // --------------------------------------- 
   //           done with case 1)
   // ---------------------------------------
      
   TEveRecTrack t;
   t.fBeta = 1.;
   if ( track.innerPosition().x()*track.px() +
	track.innerPosition().y()*track.py() < 0 )
     {
	// std::cout << "Track " << track.pt() << "\tis inward moving" << std::endl;
	// inward moving tracks
	t.fP = TEveVector(-track.px(), -track.py(), -track.pz());
	if ( track.vy()*track.innerPosition().y() > 0 )
	  t.fV = TEveVector(track.vx(), track.vy(), track.vz());
	   else
	  t.fV = TEveVector(track.vx()-track.px()/track.py()*track.vy(),
			    track.py()>0?-zeroOffset:zeroOffset,
			    track.vz()-track.pz()/track.py()*track.vy());
	t.fSign = -track.charge();
     } 
   else
     {
	// outward moving tracks
	t.fP = TEveVector(track.px(), track.py(), track.pz());
	if ( track.vy()*track.innerPosition().y() > 0 )
	  t.fV = TEveVector(track.vx(), track.vy(), track.vz());
	else
	  t.fV = TEveVector(track.vx()-track.px()/track.py()*track.vy(),
			    track.py()>0?zeroOffset:-zeroOffset,
			    track.vz()-track.pz()/track.py()*track.vy());
	t.fSign = track.charge();
     }
   TEveTrack* trk = new TEveTrack(&t,propagator);
   // now we have to make sure that the order of states is right
   if ( track.innerPosition().Rho() < track.outerPosition().Rho() ) {
      trk->AddPathMark( mark1 );
      trk->AddPathMark( mark2 );
   } else {
      // std::cout << "Track " << track.pt() << "\tstates are inward ordered" << std::endl;
      trk->AddPathMark( mark2 );
      trk->AddPathMark( mark1 );
   }
   trk->SetMainColor(color);
   // trk->MakeTrack();
   // trackList->AddElement( trk );
   result.push_back( trk );
   return result;
}
   

std::vector<TEveTrack*> 
TracksProxy3DBuilder::prepareSimpleTrack(const reco::Track& track, 
						    TEveTrackPropagator* propagator,
						    TEveElement* trackList,
						    Color_t color)
{
   std::vector<TEveTrack*> result;
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fV = TEveVector(track.vx(), track.vy(), track.vz());
   t.fP = TEveVector(track.px(), track.py(), track.pz());
   t.fSign = track.charge();
   TEveTrack* trk = new TEveTrack(&t,propagator);
   trk->SetMainColor(color);
   // trk->MakeTrack();
   // trackList->AddElement( trk );
   result.push_back( trk );
   return result;
}



REGISTER_FWRPZDATAPROXYBUILDER(TracksProxy3DBuilder,reco::TrackCollection,"Tracks");
