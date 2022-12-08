#ifndef Geom_RectangularPlaneBounds_H
#define Geom_RectangularPlaneBounds_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

/** \class RectangularPlaneBounds
 *  Rectangular plane bounds.
 *  Local Coordinate system coincides with center of the box
 *  with X axis along the width and Y axis along the lenght.
 */
class RectangularPlaneBounds final : public Bounds {
public:
  /// Construct from  half width (extension in local X),
  /// half length (Y) and half thickness (Z)
  RectangularPlaneBounds(float w, float h, float t);
  ~RectangularPlaneBounds() override;

  /// Lenght along local Y
  float length() const override { return 2 * halfLength; }
  /// Width along local X
  float width() const override { return 2 * halfWidth; }
  /// Thickness of the volume in local Z
  float thickness() const override { return 2 * halfThickness; }

  // basic bounds function
  using Bounds::inside;

  bool inside(const Local2DPoint& p) const override {
    return (std::abs(p.x()) < halfWidth) && (std::abs(p.y()) < halfLength);
  }

  bool inside(const Local3DPoint& p) const override {
    return (std::abs(p.x()) < halfWidth) && (std::abs(p.y()) < halfLength) && (std::abs(p.z()) < halfThickness);
  }

  bool inside(const Local2DPoint& p, float tollerance) const override {
    return (std::abs(p.x()) < (halfWidth + tollerance)) && (std::abs(p.y()) < (halfLength + tollerance));
  }

  bool inside(const Local3DPoint& p, const LocalError& err, float scale = 1.f) const override;

  bool inside(const Local2DPoint& p, const LocalError& err, float scale = 1.f) const override;

  float significanceInside(const Local3DPoint&, const LocalError&) const override;

  // compatible of being inside or outside...
  std::pair<bool, bool> inout(const Local3DPoint& p, const LocalError& err, float scale = 1.f) const;

  Bounds* clone() const override;

private:
  float halfWidth;
  float halfLength;
  float halfThickness;
};

#endif
