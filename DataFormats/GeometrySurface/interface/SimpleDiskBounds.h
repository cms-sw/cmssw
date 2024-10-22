#ifndef Geom_SimpleDiskBounds_H
#define Geom_SimpleDiskBounds_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

/** \class SimpleDiskBounds
 * Plane bounds that define a disk with a concentric hole in the middle.
 */

class SimpleDiskBounds final : public Bounds {
public:
  /// Construct the bounds from min and max R and Z in LOCAL coordinates.
  SimpleDiskBounds(float rmin, float rmax, float zmin, float zmax);

  float length() const override { return theZmax - theZmin; }
  float width() const override { return 2 * theRmax; }
  float thickness() const override { return theZmax - theZmin; }

  bool inside(const Local3DPoint& p) const override {
    return ((p.z() > theZmin) && (p.z() < theZmax)) &&
           ((p.perp2() > theRmin * theRmin) && (p.perp2() < theRmax * theRmax));
  }

  using Bounds::inside;

  bool inside(const Local3DPoint& p, const LocalError& err, float scale) const override;

  virtual bool inside(const Local2DPoint& p, const LocalError& err) const;

  Bounds* clone() const override;

  /// Extension of the Bounds interface
  float innerRadius() const { return theRmin; }
  float outerRadius() const { return theRmax; }

  float minZ() const { return theZmin; }
  float maxZ() const { return theZmax; }

private:
  float theRmin;
  float theRmax;
  float theZmin;
  float theZmax;
};

#endif  // Geom_SimpleDiskBounds_H
