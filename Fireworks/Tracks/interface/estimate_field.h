#ifndef Fireworks_Tracks_estimate_field_h
#define Fireworks_Tracks_estimate_field_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     estimate_field
//
/**\class estimate_field estimate_field.h Fireworks/Tracks/interface/estimate_field.h

   Description: estimate the magnetic field strength based on the curvature of a reco::Track

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Jan  6 16:15:51 EST 2009
// $Id: estimate_field.h,v 1.1 2009/01/06 21:38:41 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace reco {
   class Track;
}

namespace fw {
   double estimate_field( const reco::Track& track );
}

#endif
