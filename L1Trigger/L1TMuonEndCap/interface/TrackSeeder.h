#ifndef __L1TMUON_TRACKSEEDER_H__
#define __L1TMUON_TRACKSEEDER_H__
// 
// Class: L1TMuon::TrackSeeder
//
// Info: This class takes a list of stubs in an eta-phi slice and attempts
//       to find seeds for tracks. A track seed defines a reference point
//       from which other stubs may be found and added to the track.
//
// Author: L. Gray (FNAL)
//

#include <memory>
#include <vector>
#include <algorithm>

namespace L1TMuon {
  class TrackSeeder {
  public:
    TrackSeeder();
    ~TrackSeeder();
  };
}

#endif
