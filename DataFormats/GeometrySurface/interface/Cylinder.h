#ifndef Geom_Cylinder_H
#define Geom_Cylinder_H

/** \class Cylinder
 *
 *  A Cylinder.
 *
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() methods. 
 *  (The normal constructor will become private in the future).
 *
 *  $Date: 2012/12/26 11:05:27 $
 *  $Revision: 1.10 $
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

class Cylinder GCC11_FINAL  : public Surface {
public:

  template<typename... Args>
    Cylinder(Scalar radius, Args&& ... args) :
    Surface(std::forward<Args>(args)...), theRadius(radius){}
  
  // average Rmin Rmax...
  static float computeRadius(Bounds const & bounds) {
    return  0.5f*(bounds.width() - bounds.thickness());
  }


  typedef ReferenceCountingPointer<Cylinder> CylinderPointer;
  typedef ConstReferenceCountingPointer<Cylinder> ConstCylinderPointer;
  typedef ReferenceCountingPointer<Cylinder> BoundCylinderPointer;
  typedef ConstReferenceCountingPointer<Cylinder> ConstBoundCylinderPointer;


  /// Construct a cylinder with the specified radius.
  /// The reference frame is defined by pos and rot;
  /// the cylinder axis is parallel to the local Z axis.
  /*
  template<typename... Args>
  static CylinderPointer build(Args&& ... args) {
    return CylinderPointer(new Cylinder(std::forward<Args>(args)...));
  }
  */

  static CylinderPointer build(const PositionType& pos, const RotationType& rot,
			       Scalar radius, Bounds* bounds=0) {
    return CylinderPointer(new Cylinder(radius,pos,rot,bounds));
  }

  static CylinderPointer build(Scalar radius, const PositionType& pos, const RotationType& rot,
			       Bounds* bounds=0) {
    return CylinderPointer(new Cylinder(radius,pos,rot,bounds));
  }

  ~Cylinder(){}


  // -- Extension of Surface interface for cylinder

  /// Radius of the cylinder
  Scalar radius() const {return theRadius;}

  // -- Implementation of Surface interface    

  using Surface::side;
  virtual Side side( const LocalPoint& p, Scalar toler) const;

  /// tangent plane to surface from global point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const GlobalPoint&) const;
  /// tangent plane to surface from local point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const LocalPoint&) const;

  /// tangent plane to surface from global point
  Plane fastTangent(const GlobalPoint& aPoint) const{
    GlobalVector yPlane(rotation().z());
    GlobalVector xPlane(yPlane.cross(aPoint-position()));
    return Plane(aPoint,RotationType(xPlane, yPlane));
  }

 /// tangent plane to surface from local point
  Plane fastTangent(const LocalPoint& aPoint) const {
    return fastTangent(toGlobal(aPoint));
  } 

private:

  Scalar theRadius;

};

#endif
