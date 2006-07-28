#ifndef ReferenceHitMatcher_H
#define ReferenceHitMatcher_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/Surface/interface/LocalError.h"

#include <utility>
#include <vector>

class GluedGeomDet;

class ReferenceHitMatcher {
public:

  typedef TransientTrackingRecHit::RecHitPointer          RecHitPointer;
  typedef std::pair<bool,RecHitPointer>                   ReturnType;
  typedef TransientTrackingRecHit::ConstRecHitContainer   RecHitContainer;

  ReturnType match( const TransientTrackingRecHit& monoHit, 
		    const TransientTrackingRecHit& stereoHit,
		    const GluedGeomDet& gdet,
		    const LocalVector& dir) const;

  RecHitContainer match( const RecHitContainer& monoHits,
			 const RecHitContainer& stereoHits,
			 const GluedGeomDet& gdet,
			 const LocalVector& dir) const;

  void dumpHit(const TransientTrackingRecHit& hit) const;

private:

  LocalError weightedMean( const LocalError& a, const LocalError& b) const;

  /// project the 1D hit position (center of strip) and direction (of strip locally)
  /// onto a plane, along direction dir which is local to the plane

  std::pair<LocalPoint,LocalVector> projectHit( const TransientTrackingRecHit& hit, 
						const BoundPlane& plane,
						const LocalVector& dir) const;
  LocalPoint crossing( const std::pair<LocalPoint,LocalVector>& a,
		       const std::pair<LocalPoint,LocalVector>& b) const;

  LocalError rotateError( const TransientTrackingRecHit& hit,
			  const Plane& plane) const;

  LocalVector dloc2( const LocalError& err) const;

  double sigp2( const LocalError& err) const;

  LocalError orcaMatchedError( const TransientTrackingRecHit& monoHit, 
			       const TransientTrackingRecHit& stereoHit,
			       const GluedGeomDet& gdet) const;
};

#endif
