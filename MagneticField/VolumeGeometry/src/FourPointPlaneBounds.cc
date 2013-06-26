// #include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/FourPointPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/GeomExceptions.h"
#include <algorithm>
#include <iostream>

FourPointPlaneBounds::FourPointPlaneBounds( const LocalPoint& a, const LocalPoint& b, 
					    const LocalPoint& c, const LocalPoint& d) {
    corners_[0] = Local2DPoint( a.x(), a.y());
    corners_[1] = Local2DPoint( b.x(), b.y());
    corners_[2] = Local2DPoint( c.x(), c.y());
    corners_[3] = Local2DPoint( d.x(), d.y());

// check for convexity
    for (int i=0; i<4; ++i) {
	if (checkSide( i, corner(i+2)) *  checkSide( i, corner(i+3)) < 0) { // not on same side
	    throw GeometryError("FourPointPlaneBounds: coners not in order or not convex");
	}
    }

    double side = checkSide( 0, corners_[2]); // - for clockwise corners, + for counterclockwise
    if (side < 0) {
	std::cout << "FourPointPlaneBounds: Changing order of corners to counterclockwise" << std::endl;
	std::swap( corners_[1], corners_[3]);
    }
}

double FourPointPlaneBounds::checkSide( int i, const Local2DPoint& lp) const {
    return checkSide(i, lp.x(), lp.y());
}

bool FourPointPlaneBounds::inside( const Local3DPoint& lp) const
{
    for (int i=0; i<4; ++i) {
	if (checkSide(i,lp.x(),lp.y()) < 0) return false;
    }
    return true;
}

float FourPointPlaneBounds::length() const {return 0;}
float FourPointPlaneBounds::width() const {return 0;}
float FourPointPlaneBounds::thickness() const {return 0;}
