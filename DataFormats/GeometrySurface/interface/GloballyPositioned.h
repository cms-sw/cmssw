#ifndef Geom_GloballyPositioned_H
#define Geom_GloballyPositioned_H

#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

/** Base class for surfaces and volumes positioned in global 3D space.
 *  This class defines a cartesian reference frame, called in the 
 *  following the local frame.
 *  It provides position, orientation, and frame transformations for
 *  points and vectors.
 */

template <class T>
class GloballyPositioned {
public:
  typedef T Scalar;
  typedef Point3DBase<T, GlobalTag> PositionType;
  typedef TkRotation<T> RotationType;
  typedef Point3DBase<T, GlobalTag> GlobalPoint;
  typedef Point3DBase<T, LocalTag> LocalPoint;
  typedef Vector3DBase<T, GlobalTag> GlobalVector;
  typedef Vector3DBase<T, LocalTag> LocalVector;

  static T iniPhi() { return 999.9978; }
  static T iniEta() { return 999.9978; }

  GloballyPositioned() { setCache(); }
  GloballyPositioned(const PositionType& pos, const RotationType& rot) : thePos(pos), theRot(rot) { setCache(); }

  virtual ~GloballyPositioned() {}

  const PositionType& position() const { return thePos; }

  const RotationType& rotation() const { return theRot; }

  T phi() const { return thePhi; }
  T eta() const { return theEta; }

  // multiply inverse is faster
  class ToLocal {
  public:
    ToLocal(GloballyPositioned const& frame) : thePos(frame.position()), theRot(frame.rotation().transposed()) {}

    LocalPoint operator()(const GlobalPoint& gp) const { return toLocal(gp); }

    LocalVector operator()(const GlobalVector& gv) const { return toLocal(gv); }

    LocalPoint toLocal(const GlobalPoint& gp) const {
      return LocalPoint(theRot.multiplyInverse(gp.basicVector() - thePos.basicVector()));
    }

    LocalVector toLocal(const GlobalVector& gv) const { return LocalVector(theRot.multiplyInverse(gv.basicVector())); }

    // private:
    PositionType thePos;
    RotationType theRot;
  };

  /** Transform a local point (i.e. a point with coordinates in the
   *  local frame) to the global frame
   */
  GlobalPoint toGlobal(const LocalPoint& lp) const {
    return GlobalPoint(rotation().multiplyInverse(lp.basicVector()) + position().basicVector());
  }

  /** Transform a local point with different float precision from the
   *  one of the reference frame, and return a global point with the
   *  same precision as the input one.
   */
  template <class U>
  Point3DBase<U, GlobalTag> toGlobal(const Point3DBase<U, LocalTag>& lp) const {
    return Point3DBase<U, GlobalTag>(rotation().multiplyInverse(lp.basicVector()) + position().basicVector());
  }

  /** Transform a local vector (i.e. a vector with coordinates in the
   *  local frame) to the global frame
   */
  GlobalVector toGlobal(const LocalVector& lv) const {
    return GlobalVector(rotation().multiplyInverse(lv.basicVector()));
  }

  /** Transform a local vector with different float precision from the
   *  one of the reference frame, and return a global vector with the
   *  same precision as the input one.
   */
  template <class U>
  Vector3DBase<U, GlobalTag> toGlobal(const Vector3DBase<U, LocalTag>& lv) const {
    return Vector3DBase<U, GlobalTag>(rotation().multiplyInverse(lv.basicVector()));
  }

  /** Transform a global point (i.e. a point with coordinates in the
   *  global frame) to the local frame
   */
  LocalPoint toLocal(const GlobalPoint& gp) const {
    return LocalPoint(rotation() * (gp.basicVector() - position().basicVector()));
  }

  /** Transform a global point with different float precision from the
   *  one of the reference frame, and return a local point with the
   *  same precision as the input one.
   */
  template <class U>
  Point3DBase<U, LocalTag> toLocal(const Point3DBase<U, GlobalTag>& gp) const {
    return Point3DBase<U, LocalTag>(rotation() * (gp.basicVector() - position().basicVector()));
  }

  /** Transform a global vector (i.e. a vector with coordinates in the
   *  global frame) to the local frame
   */
  LocalVector toLocal(const GlobalVector& gv) const { return LocalVector(rotation() * gv.basicVector()); }

  /** Transform a global vector with different float precision from the
   *  one of the reference frame, and return a local vector with the
   *  same precision as the input one.
   */
  template <class U>
  Vector3DBase<U, LocalTag> toLocal(const Vector3DBase<U, GlobalTag>& gv) const {
    return Vector3DBase<U, LocalTag>(rotation() * gv.basicVector());
  }

  /** Move the position of the frame in the global frame.  
   *  Useful e.g. for alignment.
   */
  void move(const GlobalVector& displacement) {
    thePos += displacement;
    setCache();
  }

  /** Rotate the frame in the global frame.
   *  Useful e.g. for alignment.
   */
  void rotate(const RotationType& rotation) {
    theRot *= rotation;
    setCache();
  }

private:
  PositionType thePos;
  RotationType theRot;

  /*
  void resetCache() {
    if ((thePos.x() == 0.) && (thePos.y() == 0.)) {
      thePhi = theEta = 0.; // avoid FPE
    } else {
      thePhi = iniPhi();
      theEta = iniEta();
    }
  }
 */

  void setCache() {
    if ((thePos.x() == 0.) && (thePos.y() == 0.)) {
      thePhi = theEta = 0.;  // avoid FPE
    } else {
      thePhi = thePos.barePhi();
      theEta = thePos.eta();
    }
  }

  T thePhi;
  T theEta;
};

#endif
