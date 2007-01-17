

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

BoundCylinder::BoundCylinder( const PositionType& pos, 
			      const RotationType& rot, 
			      const Bounds* bounds) :
    Cylinder( pos, rot, bounds->width()/2. - bounds->thickness()/2.), 
    BoundSurface(pos, rot, bounds), Surface( pos,rot) {}

BoundCylinder::BoundCylinder( const PositionType& pos, 
			      const RotationType& rot, 
			      const Bounds& bounds) :
    Cylinder( pos, rot, bounds.width()/2. - bounds.thickness()/2.),
    BoundSurface(pos, rot, bounds), Surface( pos,rot) {}
