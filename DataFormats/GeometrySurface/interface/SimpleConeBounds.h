#ifndef Geom_SimpleConeBounds_H
#define Geom_SimpleConeBounds_H

/** \class SimpleConeBounds
 *  Cone bounds. The cone axis coincides with the Z axis.
 *  The bounds limit the length at constant Z, and allow finite thickness.
 *
 *  \warning: should be revised, probably works only when local and global
 *  Z axis coincide
 *
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include <cmath>
#include <limits>
#include <algorithm>

class SimpleConeBounds final : public Bounds {
public:
  /// Construct from inner/outer radius on the two Z faces
  SimpleConeBounds(float zmin, float rmin_zmin, float rmax_zmin, float zmax, float rmin_zmax, float rmax_zmax)
      : theZmin(zmin),
        theRminZmin(rmin_zmin),
        theRmaxZmin(rmax_zmin),
        theZmax(zmax),
        theRminZmax(rmin_zmax),
        theRmaxZmax(rmax_zmax) {
    if (theZmin > theZmax) {
      std::swap(theRminZmin, theRminZmax);
      std::swap(theRmaxZmin, theRmaxZmax);
    }
    if (theRminZmin > theRmaxZmin)
      std::swap(theRminZmin, theRmaxZmin);
    if (theRminZmax > theRmaxZmax)
      std::swap(theRminZmax, theRmaxZmax);
  }

  /// Length along Z.
  float length() const override { return theZmax - theZmin; }
  /// Maximum diameter.
  float width() const override { return 2 * std::max(theRmaxZmin, theRmaxZmax); }
  /// Thickness in the middle (Z center).
  /// Maybe it's useless, but it is pure abstract in Bounds...
  float thickness() const override { return ((theRmaxZmin - theRminZmin) + (theRmaxZmax - theRminZmax)) / 2.; }

  using Bounds::inside;

  bool inside(const Local3DPoint& p) const override {
    float lrmin = (p.z() - theZmin) * (theRminZmax - theRminZmin) / (theZmax - theZmin);
    float lrmax = (p.z() - theZmin) * (theRmaxZmax - theRmaxZmin) / (theZmax - theZmin);
    return p.z() > theZmin && p.z() < theZmax && p.perp() > lrmin && p.perp() < lrmax;
  }

  bool inside(const Local3DPoint& p, const LocalError& err, float scale) const override {
    // std::cout << "WARNING: SimpleConeBounds::inside(const Local3DPoint&, const LocalError not fully implemented"
    //	      << std::endl;     // FIXME! does not check R.
    SimpleConeBounds tmp(theZmin - sqrt(err.yy()) * scale,
                         theRminZmin,
                         theRmaxZmin,
                         theZmax + sqrt(err.yy()) * scale,
                         theRminZmax,
                         theRmaxZmax);
    return tmp.inside(p);
  }

  virtual bool inside(const Local2DPoint& p, const LocalError& err) const { return Bounds::inside(p, err); }

  Bounds* clone() const override { return new SimpleConeBounds(*this); }

  // Extension of the Bounds interface
  Geom::Theta<float> openingAngle() const {
    float theta = atan(((theRmaxZmax + theRminZmax) / 2. - (theRmaxZmin + theRminZmin) / 2.) / length());
    return Geom::Theta<float>(theta < 0 ? theta + Geom::pi() : theta);
  }

  GlobalPoint vertex() const {
    float rAtZmax = (theRmaxZmax + theRminZmax) / 2.;
    float rAtZmin = (theRmaxZmin + theRminZmin) / 2.;
    float dr = (rAtZmax - rAtZmin);

    if (std::abs(dr) < 0.0001) {  // handle degenerate case (cone->cylinder)
      return GlobalPoint(0, 0, std::numeric_limits<float>::max());
    } else {
      return GlobalPoint(0, 0, (theZmin * rAtZmax - theZmax * rAtZmin) / dr);
    }
  }

private:
  float theZmin;
  float theRminZmin;
  float theRmaxZmin;
  float theZmax;
  float theRminZmax;
  float theRmaxZmax;
};

#endif  // Geom_SimpleConeBounds_H
