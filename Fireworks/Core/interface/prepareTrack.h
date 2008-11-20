#ifndef Fireworks_Core_prepareTrack_h
#define Fireworks_Core_prepareTrack_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     prepareTrack
// 
/**\class prepareTrack prepareTrack.h Fireworks/Core/interface/prepareTrack.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 19:14:11 EST 2008
// $Id$
//

// system include files
#include "Rtypes.h"

// user include files

// forward declarations
namespace reco {
   class Track;
}
class TEveElement;
class TEveTrackPropagator;

namespace fireworks {
   TEveTrack* prepareTrack(const reco::Track& track,
                           TEveTrackPropagator* propagator,
                           TEveElement* trackList,
                           Color_t color);
}

#endif
