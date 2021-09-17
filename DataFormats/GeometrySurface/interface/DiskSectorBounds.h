#ifndef DataFormats_GeometrySurface_DiskSectorBounds_h
#define DataFormats_GeometrySurface_DiskSectorBounds_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <algorithm>
#include <cmath>
#include <cassert>

class DiskSectorBounds final : public Bounds {
public:
  DiskSectorBounds(float rmin, float rmax, float zmin, float zmax, float phiExt)
      : theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax), thePhiExtH(0.5f * phiExt) {
    assert(thePhiExtH > 0);
    if (theRmin > theRmax)
      std::swap(theRmin, theRmax);
    if (theZmin > theZmax)
      std::swap(theZmin, theZmax);
    theOffset = theRmin + 0.5f * (theRmax - theRmin);
  }

  float length() const override { return theRmax - theRmin * std::cos(thePhiExtH); }
  float width() const override { return 2.f * theRmax * std::sin(thePhiExtH); }
  float thickness() const override { return theZmax - theZmin; }

  bool inside(const Local3DPoint& p) const override;

  bool inside(const Local3DPoint& p, const LocalError& err, float scale) const override;

  Bounds* clone() const override { return new DiskSectorBounds(*this); }

  float innerRadius() const { return theRmin; }
  float outerRadius() const { return theRmax; }
  float phiHalfExtension() const { return thePhiExtH; }

private:
  float theRmin;
  float theRmax;
  float theZmin;
  float theZmax;
  float thePhiExtH;
  float theOffset;
};

#endif
