#ifndef DataFormats_TrackerRecHit2D_TrackingRecHitLessFromGlobalPosition_H
#define DataFormats_TrackerRecHit2D_TrackingRecHitLessFromGlobalPosition_H

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <functional>
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
/** Defines order of layers in the Tracker as seen by straight tracks
 *  coming from the interaction region.
 */


class TrackingRecHitLessFromGlobalPosition {
public:

  TrackingRecHitLessFromGlobalPosition( const TrackingGeometry * geometry_, PropagationDirection dir = alongMomentum) :
    geometry(geometry_), theDir(dir){  }
  
  
  bool operator()( const TrackingRecHit& a, const TrackingRecHit& b) const {
    if (theDir == alongMomentum) return insideOutLess( a, b);
    else return insideOutLess( b, a);
  }
  
 private:

  bool insideOutLess(  const TrackingRecHit& a, const TrackingRecHit& b) const;
  
  bool barrelForwardLess(  const TrackingRecHit& a, const TrackingRecHit& b) const;
  
  
  const TrackingGeometry * geometry;
  PropagationDirection theDir;
};
#endif
