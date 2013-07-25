#ifndef Fireworks_Tracks_estimate_field_h
#define Fireworks_Tracks_estimate_field_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     estimate_field
// $Id: estimate_field.h,v 1.4 2009/12/11 21:18:45 dmytro Exp $
//
namespace reco {
   class Track;
}

namespace fw {
   double estimate_field( const reco::Track& track, bool highQuality = false );
   double estimate_field( double vx1, double vy1, double vx2, double vy2, double px, double py );
}

#endif
