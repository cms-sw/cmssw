#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

BoundCylinder::BoundCylinder( const PositionType& pos, 
			      const RotationType& rot, 
			      const Bounds* bounds) :
    Surface( pos,rot ),
    Cylinder( pos, rot, bounds->width()/2. - bounds->thickness()/2.),
    BoundSurface( pos, rot, bounds ) 
{ }

BoundCylinder::BoundCylinder( const PositionType& pos, 
			      const RotationType& rot, 
			      const Bounds& bounds) :
    Surface( pos,rot ),
    Cylinder( pos, rot, bounds.width()/2. - bounds.thickness()/2.),
    BoundSurface( pos, rot, bounds )
{ }
