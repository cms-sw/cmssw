#ifndef TrackingRecHit_H
#define TrackingRecHit_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

class TrackingRecHit {
public:

  virtual ~TrackingRecHit() {}

  virtual TrackingRecHit * clone() const = 0;

  virtual AlgebraicVector parameters() const = 0;

  virtual AlgebraicSymMatrix parametersError() const = 0;
  
  virtual AlgebraicMatrix projectionMatrix() const = 0;

  virtual int dimension() const = 0;

  /// Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const = 0;

  /// Non-const access to component RecHits (if any)
  virtual std::vector<TrackingRecHit*> recHits() = 0;

  /// Access to the GeomDet (essentially the Surface, with alignment interface)
  virtual const GeomDet& det() const = 0;

  virtual DetId geographicalId() const = 0;

};

#endif
