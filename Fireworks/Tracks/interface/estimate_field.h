#ifndef Fireworks_Tracks_estimate_field_h
#define Fireworks_Tracks_estimate_field_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     estimate_field
//
namespace reco {
  class Track;
}

namespace fw {
  double estimate_field(const reco::Track& track, bool highQuality = false);
  double estimate_field(double vx1, double vy1, double vx2, double vy2, double px, double py);
}  // namespace fw

#endif
