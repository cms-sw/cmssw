// -*- C++ -*-
//
// Package:     Tracks
// Class  :     estimate_field
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jan  6 16:15:57 EST 2009
// $Id$
//

// system include files
#include "Math/Vector3D.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Tracks/interface/estimate_field.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

double
fw::estimate_field( const reco::Track& track )
{
   if ( ! track.extra().isAvailable() ) return -1;
   math::XYZVector displacement(track.outerPosition().x()-track.innerPosition().x(),
				track.outerPosition().y()-track.innerPosition().y(),
				0);
   math::XYZVector transverseMomentum(track.innerMomentum().x(),
				      track.innerMomentum().y(),
				      0);
   double cosAlpha = transverseMomentum.Dot(displacement)/transverseMomentum.r()/displacement.r();
   return 200*sqrt(1-cosAlpha*cosAlpha)/0.2998*transverseMomentum.r()/displacement.r();
}

