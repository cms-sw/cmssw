#ifndef RECO_MAP_HIT_TSOS_H
#define RECO_MAP_HIT_TSOS_H

#include <map>

class TrackingRecHit;
class TrajectoryStateOnSurface;

namespace reco {
  /** 
  * @brief 
  *   Map of Tracking RecHit <-> TSOS association
  *   [Note: Pointers to Tracking RectHit should be taken from Track]
  */
  typedef std::map<const TrackingRecHit *, TrajectoryStateOnSurface> MHitTSOS;
}

#endif // RECO_MAP_HIT_TSOS_H
