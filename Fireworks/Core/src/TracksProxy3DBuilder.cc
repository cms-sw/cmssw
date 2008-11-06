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
// $Id: TracksProxy3DBuilder.cc,v 1.19 2008/09/26 22:00:32 dmytro Exp $
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

       TEveTrack* trk = prepareTrack( *it, propagator, trkList, iItem->defaultDisplayProperties().color() );
       trk->MakeTrack();
       trkList->AddElement( trk );

       gEve->AddElement(trkList,tList);
      // printf("track pt: %.1f, eta: %0.1f => B: %0.2f T\n", it->pt(), it->eta(), fw::estimate_field(*it));
    }
}

TEveTrack*
TracksProxy3DBuilder::prepareSimpleTrack(const reco::Track& track,
						    TEveTrackPropagator* propagator,
						    TEveElement* trackList,
						    Color_t color)
{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fV = TEveVector(track.vx(), track.vy(), track.vz());
   t.fP = TEveVector(track.px(), track.py(), track.pz());
   t.fSign = track.charge();
   TEveTrack* trk = new TEveTrack(&t,propagator);
   trk->SetMainColor(color);
   return trk;
}

TEveTrack*
TracksProxy3DBuilder::prepareTrack(const reco::Track& track,
				      TEveTrackPropagator* propagator,
				      TEveElement* trackList,
				      Color_t color)
{
   // To make use of all available information, we have to order states
   // properly first. Propagator should take care of y=0 transition.

   if ( ! track.extra().isAvailable() )
     return prepareSimpleTrack(track,propagator,trackList,color);

   // we have 3 states for sure, bust some of them may overlap.
   // POCA can be either initial point of trajector if we deal
   // with normal track or just one more state. So we need first
   // to decide where is the origin of the track.

   bool outsideIn = ( track.innerPosition().x()*track.innerMomentum().x()+
		      track.innerPosition().y()*track.outerMomentum().y() < 0 );

   TEveRecTrack t;
   t.fBeta = 1.;
   t.fSign = track.charge();

   if ( outsideIn ) {
      t.fV = TEveVector( track.innerPosition().x(),
			 track.innerPosition().y(),
			 track.innerPosition().z() );
      t.fP = TEveVector( track.innerMomentum().x(),
			 track.innerMomentum().y(),
			 track.innerMomentum().z() );
   } else {
      t.fV = TEveVector( track.vertex().x(),
			 track.vertex().y(),
			 track.vertex().z() );
      t.fP = TEveVector( track.px(),
			 track.py(),
			 track.pz() );
   }

   TEveTrack* trk = new TEveTrack(&t,propagator);
   if ( outsideIn )
     trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
   trk->SetMainColor(color);

   if ( !outsideIn ) {
      TEvePathMark mark( TEvePathMark::kDaughter );
      mark.fV = TEveVector( track.innerPosition().x(),
			    track.innerPosition().y(),
			    track.innerPosition().z() );
      trk->AddPathMark( mark );
   }

   TEvePathMark mark1( TEvePathMark::kDecay );
   mark1.fV = TEveVector( track.outerPosition().x(),
			  track.outerPosition().y(),
			  track.outerPosition().z() );

   trk->AddPathMark( mark1 );
   return trk;
}

REGISTER_FWRPZDATAPROXYBUILDER(TracksProxy3DBuilder,reco::TrackCollection,"Tracks");
