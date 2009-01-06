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
// $Id: prepareTrack.cc,v 1.2 2008/12/04 15:28:59 dmytro Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

// user include files
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Fireworks/Candidates/interface/prepareSimpleTrack.h"


//
// constants, enums and typedefs
//
//
// static data member definitions
//
namespace fireworks {
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
}
