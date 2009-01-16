#ifndef Fireworks_Tracks_TracksRecHitsUtil_h
#define Fireworks_Tracks_TracksRecHitsUtil_h


// -*- C++ -*-
// $Id: TracksRecHitsUtil.h,v 1.2 2009/01/16 10:37:00 amraktad
//

#include "DataFormats/TrackReco/interface/Track.h"

class FWEventItem;
class TEveElementList;
class TEveElement;

class TracksRecHitsUtil
{
public:
   static void buildTracksRecHits(const FWEventItem* iItem, TEveElementList** product);
   static void addHits(const reco::Track& track,
                       const FWEventItem* iItem,
                       TEveElement* trkList);
};

#endif
