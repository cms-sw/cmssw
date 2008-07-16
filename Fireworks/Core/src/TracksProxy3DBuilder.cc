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
// $Id: TracksProxy3DBuilder.cc,v 1.11 2008/07/07 06:14:02 dmytro Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
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
    std::cout <<"build called"<<std::endl;

    //since we created it, we know the type (would like to do this better)
    TEveTrackList* tlist = dynamic_cast<TEveTrackList*>(*product);
    if ( !tlist && *product ) {
       std::cout << "incorrect type" << std::endl;
       return;
    }
       
    if(0 == tlist) {
      tlist =  new TEveTrackList(iItem->name().c_str());
      *product = tlist;
      tlist->SetMainColor(iItem->defaultDisplayProperties().color());
      TEveTrackPropagator* propagator = tlist->GetPropagator();
      //units are Tesla
      propagator->SetMagField( -4.0);
      //get this from geometry, units are CM
      propagator->SetMaxR(123.0);
      propagator->SetMaxZ(300.0);
      
      gEve->AddElement(tlist);
    } else {
      tlist->DestroyElements();
    }

    const reco::TrackCollection* tracks=0;
    iItem->get(tracks);
   
    //fwlite::Handle<reco::TrackCollection> tracks;
    //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
    
    if(0 == tracks ) {
      std::cout <<"failed to get Tracks"<<std::endl;
      return;
    }
   
    
    TEveTrackPropagator* propagator = tlist->GetPropagator();
    
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
   
    int index=0;
    //cout <<"----"<<endl;
    TEveRecTrack t;

    t.fBeta = 1.;
    for(reco::TrackCollection::const_iterator it = tracks->begin();
	it != tracks->end();++it,++index) {
       // use extra information if available
       t.fP = TEveVector(it->px(), it->py(), it->pz());
       t.fV = TEveVector(it->vx(), it->vy(), it->vz());
       t.fSign = it->charge();
       
       TEveTrack* trk = new TEveTrack(&t,propagator);
       char s[1024];
       sprintf(s,"Track %d, Pt: %0.1f GeV",index,it->pt());
       trk->SetTitle(s);
       trk->SetMainColor(iItem->defaultDisplayProperties().color());
       trk->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
       trk->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
       
       if ( it->extra().isAvailable() ) {
	  TEvePathMark mark( TEvePathMark::kDaughter );
	  mark.fV = TEveVector( it->outerPosition().x(), it->outerPosition().y(), it->outerPosition().z() );
	  trk->AddPathMark( mark );
       }
       trk->MakeTrack();

       gEve->AddElement(trk,tlist);
      // printf("track pt: %.1f, eta: %0.1f => B: %0.2f T\n", it->pt(), it->eta(), fw::estimate_field(*it));
    }
    
}


REGISTER_FWRPZDATAPROXYBUILDER(TracksProxy3DBuilder,reco::TrackCollection,"Tracks");
