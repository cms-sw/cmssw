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
 *  $Date: 2012/12/23 18:07:15 $
 *  $Revision: 1.4 $
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"

class Cone  GCC11_FINAL  : public Surface {
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

  virtual Side side( const LocalPoint& p, Scalar tolerance) const {return side( toGlobal(p), tolerance);}
  virtual Side side( const GlobalPoint& p, Scalar tolerance) const;

  // Tangent plane to surface from global point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const GlobalPoint&) const;
  // Tangent plane to surface from local point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const LocalPoint&) const;


private:

  GlobalPoint             theVertex;
  Geom::Theta<Scalar>     theAngle;

};

#endif
