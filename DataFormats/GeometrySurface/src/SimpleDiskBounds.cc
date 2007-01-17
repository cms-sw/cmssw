

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

bool SimpleDiskBounds::inside( const Local3DPoint& p, const LocalError& err, float scale) const 
{
    if (p.z() < theZmin || p.z() > theZmax) return false; // check the easy part first

    double perp2 = p.perp2();
    double perp = sqrt(perp2);
    if (perp2 == 0) return scale*scale*(err.xx() + err.xy())  > theRmin*theRmin;

    // rotated error along p.x(),p.y()
    // equivalent to (but faster than) err.rotate(p.x(),p.y()).xx()
    // since we don't need all matrix elements
    float deltaR = scale * sqrt( p.x()*p.x()/perp2 * err.xx() - 
				 2*p.x()*p.y()/perp2 * err.xy() + 
				 p.y()*p.y()/perp2 * err.yy());
    return perp > std::max(theRmin-deltaR, 0.f) && perp < theRmax+deltaR;
}
