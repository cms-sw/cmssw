// Helper function for output of Track Hits info that involves next data:
//   - Track inner/outer DetId
//   - Hits
//
// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 03/29/07
// Licence: GPL

#ifndef TRACK_OSTREAM_H
#define TRACK_OSTREAM_H

#include <iosfwd>

namespace reco {
  class Track;
}

struct TrackOstream {
  explicit TrackOstream( const reco::Track &roTRACK_ORIG):
    roTRACK( roTRACK_ORIG) {}

  const reco::Track &roTRACK;
};

std::ostream &operator<< ( std::ostream &roOut, const TrackOstream &roTO);

#endif // TRACK_OSTREAM_H
