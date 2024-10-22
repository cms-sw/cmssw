#ifndef Geom_Surface_H
#define Geom_Surface_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "FWCore/Utilities/interface/clone_ptr.h"
#include <algorithm>

/** Collection of enums to specify orientation of the surface wrt the
 *  volume it a bound of.
 */
namespace SurfaceOrientation {
  enum Side { positiveSide, negativeSide, onSurface };
  enum GlobalFace { outer, inner, zplus, zminus, phiplus, phiminus };
}  // namespace SurfaceOrientation

//template <class T> class ReferenceCountingPointer;

class Plane;
using TangentPlane = Plane;

/** Base class for 2D surfaces in 3D space.
 *  May have MediumProperties.
 * may have bounds
 *  The Bounds define a region AROUND the surface.
 *  Surfaces which differ only by the shape of their bounds are of the
 *  same "surface" type  
 *  (e.g. Plane or Cylinder).
 */

class Surface : public ReferenceCountedInConditions, public GloballyPositioned<float> {
public:
  using Side = SurfaceOrientation::Side;

  using Base = GloballyPositioned<float>;

  ~Surface() override {}

protected:
  Surface() {}
  Surface(const PositionType& pos, const RotationType& rot) : Base(pos, rot) {}

  Surface(const PositionType& pos, const RotationType& rot, Bounds* bounds) : Base(pos, rot), theBounds(bounds) {}

  Surface(const PositionType& pos, const RotationType& rot, MediumProperties mp)
      : Base(pos, rot), theMediumProperties(mp) {}

  Surface(const PositionType& pos, const RotationType& rot, MediumProperties mp, Bounds* bounds)
      : Base(pos, rot), theMediumProperties(mp), theBounds(bounds) {}

  Surface(const Surface& iSurface)
      : ReferenceCountedInConditions(iSurface),
        Base(iSurface),
        theMediumProperties(iSurface.theMediumProperties),
        theBounds(iSurface.theBounds) {}

  Surface(Surface&& iSurface)
      : ReferenceCountedInConditions(iSurface),
        Base(iSurface),
        theMediumProperties(iSurface.theMediumProperties),
        theBounds(std::move(iSurface.theBounds)) {}

public:
  /** Returns the side of the surface on which the point is.
   *  Not defined for 1-sided surfaces (Moebius leaf etc.)
   *  For normal 2-sided surfaces the meaning of side is surface type dependent.
   */
  virtual Side side(const LocalPoint& p, Scalar tolerance = 0) const = 0;
  virtual Side side(const GlobalPoint& p, Scalar tolerance = 0) const { return side(toLocal(p), tolerance); }

  using Base::toGlobal;
  using Base::toLocal;

  GlobalPoint toGlobal(const Point2DBase<Scalar, LocalTag> lp) const {
    return GlobalPoint(rotation().multiplyInverse(lp.basicVector()) + position().basicVector());
  }

  const MediumProperties& mediumProperties() const { return theMediumProperties; }

  void setMediumProperties(const MediumProperties& mp) { theMediumProperties = mp; }

  const Bounds& bounds() const { return *theBounds; }

  // here and not in plane because of PixelBarrelLayer::overlap
  std::pair<float, float> const& phiSpan() const { return bounds().phiSpan(); }
  std::pair<float, float> const& zSpan() const { return bounds().zSpan(); }
  std::pair<float, float> const& rSpan() const { return bounds().rSpan(); }

  /** Tangent plane to surface from global point.
   * Returns a plane, tangent to the Surface at a point.
   * The point must be on the surface.
   * The return type is a ReferenceCountingPointer, so the plane 
   * will be deleted automatically when no longer needed.
   */
  virtual ConstReferenceCountingPointer<TangentPlane> tangentPlane(const GlobalPoint&) const = 0;
  /** Tangent plane to surface from local point.
   */
  virtual ConstReferenceCountingPointer<TangentPlane> tangentPlane(const LocalPoint&) const = 0;

protected:
  MediumProperties theMediumProperties;
  extstd::clone_ptr<Bounds> theBounds;
};

#endif  // Geom_Surface_H
