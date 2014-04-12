#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"


ReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const GlobalPoint&) const
{
  return ReferenceCountingPointer<TangentPlane>(const_cast<Plane*>(this));
}

ReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const LocalPoint&) const
{
  return ReferenceCountingPointer<TangentPlane>(const_cast<Plane*>(this));
}
