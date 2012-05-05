#ifndef Geom_SimpleDiskBounds_H
#define Geom_SimpleDiskBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"


/** \class SimpleDiskBounds
 * Plane bounds that define a disk with a concentric hole in the middle.
 */

class SimpleDiskBounds GCC11_FINAL : public Bounds {
public:

  /// Construct the bounds from min and max R and Z in LOCAL coordinates.
  SimpleDiskBounds( float rmin, float rmax, float zmin, float zmax);

  virtual float length()    const { return theZmax - theZmin;}
  virtual float width()     const { return 2*theRmax;}
  virtual float thickness() const { return theZmax-theZmin;}

  virtual bool inside( const Local3DPoint& p) const;
    
  virtual bool inside( const Local3DPoint& p, const LocalError& err, float scale) const;

  virtual bool inside( const Local2DPoint& p, const LocalError& err) const;

  virtual Bounds* clone() const;

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


