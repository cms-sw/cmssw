// -*- C++ -*-
// $Id: TracksRecHitsProxyPlain3DBuilder.cc,v 1.0 2008/02/22 10:37:00 Tom Danielson
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "RVersion.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
// include file for the RecHits
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
// include file for the TrajectorySeeds
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "TEveGeoNode.h"
#include "Fireworks/Tracks/interface/TracksRecHitsProxy3DBuilder.h"
#include "Fireworks/Tracks/interface/TracksRecHitsProxyPlain3DBuilder.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
  
void TracksRecHitsProxyPlain3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
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
     TracksRecHitsProxy3DBuilder::addHits(*it, iItem, trkList);
  }
   
}

REGISTER_FW3DDATAPROXYBUILDER(TracksRecHitsProxyPlain3DBuilder,reco::TrackCollection,"TrackHits");
