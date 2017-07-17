#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"


ConstReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const GlobalPoint&) const
{
  return ConstReferenceCountingPointer<TangentPlane>(this);
}

ConstReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const LocalPoint&) const
{
  return ConstReferenceCountingPointer<TangentPlane>(this);
}
