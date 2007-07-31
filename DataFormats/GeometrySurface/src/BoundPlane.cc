

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"


BoundPlane::BoundPlane( const PositionType& pos, 
			const RotationType& rot) :
    Plane( pos, rot), BoundSurface(pos, rot, OpenBounds()), Surface( pos,rot) {}

