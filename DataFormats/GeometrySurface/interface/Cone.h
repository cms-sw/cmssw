#ifndef Geom_Cone_H
#define Geom_Cone_H

/** \class Cone
 *
 *  A Cone.
 *
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructors will become private in the future).
 *
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"

class Cone  final  : public Surface {
public:

  template<typename... Args>
  Cone(const PositionType& vert,
       Geom::Theta<Scalar> angle, 
       Args&& ... args) :
    Surface(std::forward<Args>(args)...), 
    theVertex(vert), theAngle(angle) {}

  typedef ReferenceCountingPointer<Cone> ConePointer;
  typedef ReferenceCountingPointer<Cone> ConstConePointer;


  /// Construct a cone with the specified vertex and opening angle.
  /// The reference frame is defined by pos and rot;
  /// the cone axis is parallel to the local Z axis.
  static ConePointer build(const PositionType& pos,
			   const RotationType& rot,
			   const PositionType& vert,
			   Geom::Theta<Scalar> angle) {
    return ConePointer(new Cone(vert, angle, pos, rot));
  }


  // -- DEPRECATED CONSTRUCTOR

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  Cone( const PositionType& pos, const RotationType& rot, 
	const PositionType& vert, Geom::Theta<Scalar> angle) :
    Surface( pos, rot), theVertex(vert), theAngle(angle) {}


  // -- Extension of Surface interface for cone

  /// Global position of the cone vertex
  GlobalPoint vertex() const {return theVertex;}

  /// Angle of the cone
  Geom::Theta<float> openingAngle() const {return theAngle;}


  // -- Implementation of Surface interface    

  Side side( const LocalPoint& p, Scalar tolerance) const override {return side( toGlobal(p), tolerance);}
  Side side( const GlobalPoint& p, Scalar tolerance) const override;

  // Tangent plane to surface from global point
  ConstReferenceCountingPointer<TangentPlane> tangentPlane (const GlobalPoint&) const override;
  // Tangent plane to surface from local point
  ConstReferenceCountingPointer<TangentPlane> tangentPlane (const LocalPoint&) const override;


private:

  GlobalPoint             theVertex;
  Geom::Theta<Scalar>     theAngle;

};

#endif
