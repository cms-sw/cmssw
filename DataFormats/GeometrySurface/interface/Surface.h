#ifndef Geom_Surface_H
#define Geom_Surface_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"


#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

/*
#include "DataFormats/GeometrySurface/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
*/
/** Collection of enums to specify orientation of the surface wrt the
 *  volume it a bound of.
 */
namespace SurfaceOrientation {
  enum Side {positiveSide, negativeSide, onSurface};
  enum GlobalFace {outer,inner,zplus,zminus,phiplus,phiminus};
}


//template <class T> class ReferenceCountingPointer;

class TangentPlane;

/** Base class for 2D surfaces in 3D space.
 *  May have MediumProperties.
 */

class Surface : public GloballyPositioned<float> 
		, public ReferenceCountedInConditions 
{
public:
  typedef SurfaceOrientation::Side Side;

  typedef GloballyPositioned<float>       Base;

  Surface( const PositionType& pos, const RotationType& rot) :
    Base( pos, rot), theMediumProperties(0.,0.), m_mpSet(false) {}

  Surface( const PositionType& pos, const RotationType& rot, 
	   MediumProperties* mp) : 
    Base( pos, rot), 
    theMediumProperties(mp? *mp : MediumProperties(0.,0.)),
    m_mpSet(mp)
  {}
 
 Surface( const PositionType& pos, const RotationType& rot,
           MediumProperties mp) :
    Base( pos, rot),
    theMediumProperties(mp),
    m_mpSet(true)
  {}
 
  Surface( const Surface& iSurface ) : 
  Base( iSurface), 
  theMediumProperties(iSurface.theMediumProperties),
  m_mpSet(iSurface.m_mpSet)
  {}

  // pure virtual destructor - makes base classs abstract
  virtual ~Surface() = 0;

  /** Returns the side of the surface on which the point is.
   *  Not defined for 1-sided surfaces (Moebius leaf etc.)
   *  For normal 2-sided surfaces the meaning of side is surface type dependent.
   */
  virtual Side side( const LocalPoint& p, Scalar tolerance=0) const = 0;
  virtual Side side( const GlobalPoint& p, Scalar tolerance=0) const {
    return side( toLocal(p), tolerance);
  }

  using Base::toGlobal;
  using Base::toLocal;

  GlobalPoint toGlobal( const Point2DBase< Scalar, LocalTag> lp) const {
    return GlobalPoint( rotation().multiplyInverse( lp.basicVector()) +
			position().basicVector());
  }

  /*
  GlobalError toGlobal( const LocalError& le) const {
    return rotation().transform(le);
  }

  LocalError toLocal( const GlobalError& ge) const {
    return rotation().transform(ge);
  }
  */

  const MediumProperties* mediumProperties() const { 
    return  m_mpSet ? &theMediumProperties : 0;
  }

  void setMediumProperties( const MediumProperties & mp ) {
    theMediumProperties = mp;
    m_mpSet = true;
  }

  void setMediumProperties( MediumProperties* mp ) {
    if (mp) {
      theMediumProperties = *mp;
      m_mpSet = true;
    } else {
      theMediumProperties = MediumProperties(0.,0.);
      m_mpSet = false;
    }
  }

  /** Tangent plane to surface from global point.
   * Returns a new plane, tangent to the Surface at a point.
   * The point must be on the surface.
   * The return type is a ReferenceCountingPointer, so the plane 
   * will be deleted automatically when no longer needed.
   */
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const GlobalPoint&) const = 0;
  /** Tangent plane to surface from local point.
   */
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const LocalPoint&) const = 0;

private:

  MediumProperties theMediumProperties;
  bool m_mpSet;
};
  
inline Surface::~Surface() {}

#endif // Geom_Surface_H
