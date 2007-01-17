

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"



ReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const GlobalPoint&) const
{
  return ReferenceCountingPointer<TangentPlane>(new TangentPlane(position(),
								 rotation(),
								 this));
}

ReferenceCountingPointer<TangentPlane> 
Plane::tangentPlane (const LocalPoint&) const
{
  return ReferenceCountingPointer<TangentPlane>(new TangentPlane(position(),
								 rotation(),
								 this));
}
