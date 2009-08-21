#ifndef Fireworks_Tracks_prepareTrack_h
#define Fireworks_Tracks_prepareTrack_h
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
// $Id: prepareTrack.h,v 1.2.10.1 2009/08/20 11:38:44 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "TEveVSDStructs.h"

// forward declarations
namespace reco {
   class Track;
}
class TEveElement;
class TEveTrackPropagator;

namespace fireworks {
  struct State{
    TEveVector position;
    TEveVector momentum;
    bool valid;
  State():valid(false){}
  State(const TEveVector& pos):
    position(pos),valid(false){}
  State(const TEveVector& pos, const TEveVector& mom):
    position(pos),momentum(mom),valid(true){}
  };

  class StateOrdering{
    TEveVector m_direction;
  public:
    StateOrdering( const TEveVector& momentum ){
      m_direction = momentum;
      m_direction.Normalize();
    }
    bool operator() (const State& state1,
		     const State& state2 ) const
    {
      double product1 = state1.position.Perp()*(state1.position.fX*m_direction.fX + state1.position.fY*m_direction.fY>0?1:-1);
      double product2 = state2.position.Perp()*(state2.position.fX*m_direction.fX + state2.position.fY*m_direction.fY>0?1:-1);
      return product1 < product2;
    }
  };
   
  TEveTrack* prepareTrack(const reco::Track& track,
			  TEveTrackPropagator* propagator,
			  TEveElement* trackList,
			  Color_t color,
			  const std::vector<TEveVector>& extraRefPoints = std::vector<TEveVector>());
}

#endif
