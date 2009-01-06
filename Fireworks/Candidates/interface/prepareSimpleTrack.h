#ifndef Fireworks_Candidates_prepareSimpleTrack_h
#define Fireworks_Candidates_prepareSimpleTrack_h
// -*- C++ -*-
//
// Package:     Candidates
// Class  :     prepareTrack
// 
/**\class prepareTrack prepareTrack.h Fireworks/Candidate/interface/prepareSimpleTrack.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 19:14:11 EST 2008
// $Id: prepareTrack.h,v 1.2 2008/12/04 15:28:59 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files

// forward declarations
namespace reco {
   class Candidate;
}
class TEveElement;
class TEveTrackPropagator;

namespace fireworks {
   TEveTrack* prepareSimpleTrack(const reco::Candidate& track,
				 TEveTrackPropagator* propagator,
				 TEveElement* trackList,
				 Color_t color);
}

#endif
