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
// $Id: TracksProxy3DBuilder.cc,v 1.22 2008/11/26 02:17:24 chrjones Exp $
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

#include "Fireworks/Core/interface/prepareTrack.h"


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

       TEveTrack* trk = fireworks::prepareTrack( *it, propagator, trkList, iItem->defaultDisplayProperties().color() );
       trk->MakeTrack();
       trkList->AddElement( trk );

       gEve->AddElement(trkList,tList);
      // printf("track pt: %.1f, eta: %0.1f => B: %0.2f T\n", it->pt(), it->eta(), fw::estimate_field(*it));
    }
}

//REGISTER_FWRPZDATAPROXYBUILDERBASE(TracksProxy3DBuilder,reco::TrackCollection,"Tracks");
