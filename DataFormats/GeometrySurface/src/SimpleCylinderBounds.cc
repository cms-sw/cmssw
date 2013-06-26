#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"


SimpleCylinderBounds::SimpleCylinderBounds( float rmin, float rmax, float zmin, float zmax) : 
  theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax) {
  if ( theRmin > theRmax) std::swap( theRmin, theRmax);
  if ( theZmin > theZmax) std::swap( theZmin, theZmax);
}

bool SimpleCylinderBounds::inside( const Local3DPoint& p) const {
  return p.z()    > theZmin && p.z()    < theZmax &&
    p.perp() > theRmin && p.perp() < theRmax;
}

bool SimpleCylinderBounds::inside( const Local3DPoint& p, const LocalError& err,float scale) const {

    SimpleCylinderBounds tmp( theRmin, theRmax,
			      theZmin - sqrt(err.yy())*scale,
			      theZmax + sqrt(err.yy())*scale);
    
    return tmp.inside(p);
  }

bool SimpleCylinderBounds::inside( const Local2DPoint& p, const LocalError& err) const {
  return Bounds::inside(p,err);
}

Bounds* SimpleCylinderBounds::clone() const { 
  return new SimpleCylinderBounds(*this);
}
