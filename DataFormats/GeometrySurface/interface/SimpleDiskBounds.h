#ifndef Geom_SimpleDiskBounds_H
#define Geom_SimpleDiskBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <algorithm>
#include <cmath>

/** \class SimpleDiskBounds
 * Plane bounds that define a disk with a concentric hole in the middle.
 */

class SimpleDiskBounds : public Bounds {
public:

  /// Construct the bounds from min and max R and Z in LOCAL coordinates.
  SimpleDiskBounds( float rmin, float rmax, float zmin, float zmax) : 
    theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax) {
    if ( theRmin > theRmax) std::swap( theRmin, theRmax);
    if ( theZmin > theZmax) std::swap( theZmin, theZmax);
  }

  virtual float length()    const { return theZmax - theZmin;}
  virtual float width()     const { return 2*theRmax;}
  virtual float thickness() const { return theZmax-theZmin;}

  virtual bool inside( const Local3DPoint& p) const {
    return p.z()    > theZmin && p.z()    < theZmax &&
           p.perp() > theRmin && p.perp() < theRmax;
  }
    
  virtual bool inside( const Local3DPoint& p, const LocalError& err, float scale) const;

  virtual bool inside( const Local2DPoint& p, const LocalError& err) const {
    return Bounds::inside(p,err);
  }

  virtual Bounds* clone() const { 
    return new SimpleDiskBounds(*this);
  }

  /// Extension of the Bounds interface
  float innerRadius() const {return theRmin;}
  float outerRadius() const {return theRmax;}

private:
  float theRmin;
  float theRmax;
  float theZmin;
  float theZmax;
};

#endif // Geom_SimpleDiskBounds_H


