// -*- C++ -*-
//
// Package:     Core
// Class  :     prepareTrack
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 19:14:22 EST 2008
// $Id: prepareTrack.cc,v 1.1 2008/11/20 01:10:25 chrjones Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

// user include files
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Fireworks/Core/interface/prepareTrack.h"


//
// constants, enums and typedefs
//
//
// static data member definitions
//
namespace fireworks {
   TEveTrack*
   prepareSimpleTrack(const reco::Track& track,
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
   prepareSimpleTrack(const reco::Candidate& track,
                      TEveTrackPropagator* propagator,
                      TEveElement* trackList,
                      Color_t color)
   {
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fP = TEveVector( track.px(), track.py(), track.pz() );
      t.fV = TEveVector( track.vertex().x(), track.vertex().y(), track.vertex().z() );
      t.fSign = track.charge();
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetMainColor(color);
      return trk;
   }


TEveTrack*
prepareTrack(const reco::Track& track,
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
}
