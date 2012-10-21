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
 *  $Date: 2010/12/22 11:06:42 $
 *  $Revision: 1.6 $
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

class Cylinder : public virtual Surface {
public:
  typedef ReferenceCountingPointer<Cylinder> CylinderPointer;
  typedef ConstReferenceCountingPointer<Cylinder> ConstCylinderPointer;


  /// Construct a cylinder with the specified radius.
  /// The reference frame is defined by pos and rot;
  /// the cylinder axis is parallel to the local Z axis.
  static CylinderPointer build(const PositionType& pos,
			       const RotationType& rot,
			       Scalar radius,
			       MediumProperties* mp=0) {
    return CylinderPointer(new Cylinder(pos, rot, radius, mp));
  }


  ~Cylinder(){}

  // -- DEPRECATED CONSTRUCTORS

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  Cylinder( const PositionType& pos, const RotationType& rot, Scalar radius) :
    Surface( pos, rot), theRadius(radius) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  Cylinder( const PositionType& pos, const RotationType& rot, Scalar radius,
	 MediumProperties* mp) : 
    Surface( pos, rot, mp), theRadius(radius) {}

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
