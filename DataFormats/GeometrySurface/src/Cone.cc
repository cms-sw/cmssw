

#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

#include <iostream>

ConstReferenceCountingPointer<TangentPlane> Cone::tangentPlane (const GlobalPoint&) const {
  // FIXME: to be implemented...
  std::cout << "*** WARNING: Cone::tangentPlane not implemented." <<std::endl;
  abort();
  return ConstReferenceCountingPointer<TangentPlane>();
}

ConstReferenceCountingPointer<TangentPlane> Cone::tangentPlane (const LocalPoint&) const {
  // FIXME: to be implemented...
  std::cout << "*** WARNING: Cone::tangentPlane not implemented." <<std::endl;
  abort();
  return ConstReferenceCountingPointer<TangentPlane>();
}

Surface::Side Cone::side( const GlobalPoint& p, Scalar tolerance) const {
  // FIXME: should be done in local coordinates as this is not correct in the case the verstex is not on the (global) Z axis!!!!

  // tolerance is interpreted as max distance from cone surface.
  // FIXME: check case when vertex().z()==inf.
  GlobalPoint p1(p.x(), p.y(), p.z()-vertex().z());

  // handle the singularity of p=vertex (i.e. p1.mag() undefined)
  if (p1.mag()<tolerance) return SurfaceOrientation::onSurface;
  double delta = double(p1.theta())- double(openingAngle());
  if (fabs(delta) < tolerance/p1.mag()) return SurfaceOrientation::onSurface;
  
  if (p1.theta() < Geom::pi()/2.) {
    return (delta>0. ?  SurfaceOrientation::positiveSide : SurfaceOrientation::negativeSide);
  } else {
    return (delta>0. ?  SurfaceOrientation::negativeSide : SurfaceOrientation::positiveSide);
  }
}
