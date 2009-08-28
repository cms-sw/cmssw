#ifndef Fireworks_Tracks_estimate_field_h
#define Fireworks_Tracks_estimate_field_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     estimate_field
// $Id: estimate_field.h,v 1.2 2009/01/23 21:35:46 amraktad Exp $
//
namespace reco {
   class Track;
}

namespace fw {
   double estimate_field( const reco::Track& track );
   double estimate_field( double vx1, double vy1, double vx2, double vy2, double px, double py );
}

#endif
