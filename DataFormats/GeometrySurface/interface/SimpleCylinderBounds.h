#ifndef Geom_SimpleCylinderBounds_H
#define Geom_SimpleCylinderBounds_H

/** \class SimpleCylinderBounds
 *
 *  Cylinder bounds. The cylinder axis coincides with the Z axis.
 *  The bounds limit the length at constant Z, and allow finite thickness.
 *  The cylinder bound in this way looks like a pipe cut 
 *  perpendicularily to it's axis. Width is intended as the (outer) diameter
 *  of the pipe and thickness as the thickness of the pipe, i.e.
 *  difference between outer and inner radius
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <cmath>
#include <algorithm>

class SimpleCylinderBounds final : public Bounds {
public:
  SimpleCylinderBounds(float rmin, float rmax, float zmin, float zmax);

  /// Lenght of the cylinder
  float length() const override { return theZmax - theZmin; }
  /// Outer diameter of the cylinder
  float width() const override { return 2 * theRmax; }
  /// Thikness of the "pipe", i.e. difference between outer and inner radius
  float thickness() const override { return theRmax - theRmin; }

  using Bounds::inside;
  bool inside(const Local3DPoint& p) const override;

  bool inside(const Local3DPoint& p, const LocalError& err, float scale) const override;

  virtual bool inside(const Local2DPoint& p, const LocalError& err) const;

  Bounds* clone() const override;

private:
  float theRmin;
  float theRmax;
  float theZmin;
  float theZmax;
};

#endif  // Geom_SimpleCylinderBounds_H
