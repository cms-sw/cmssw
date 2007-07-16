#ifndef Geom_BoundSurface_H
#define Geom_BoundSurface_H


/** \class BoundSurface
 *
 *  Adds Bounds to Surface. 
 *
 *  The Bounds define a region AROUND the surface.
 *  Surfaces which differ only by the shape of their bounds are of the
 *  same "surface" type  
 *  (e.g. Plane or Cylinder).
 */


#include <memory>
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "DataFormats/GeometrySurface/interface/Bounds.h"


class BoundSurface : public virtual Surface {
public:

  BoundSurface( const PositionType& pos, 
		const RotationType& rot, 
		const Bounds* bounds) :
    Surface( pos, rot),  m_phiSpan(0.,0.),
    theBounds( bounds->clone()) { computePhiSpan();}

  BoundSurface( const PositionType& pos, 
		const RotationType& rot, 
		const Bounds& bounds) :
    Surface( pos, rot),  m_phiSpan(0.,0.),
    theBounds( bounds.clone()) { computePhiSpan();}

  BoundSurface( const PositionType& pos, 
		const RotationType& rot, 
		const Bounds* bounds, 
		MediumProperties* mp) :
    Surface( pos, rot, mp),  m_phiSpan(0.,0.),
    theBounds( bounds->clone()) { computePhiSpan();}

  BoundSurface( const PositionType& pos, 
		const RotationType& rot, 
		const Bounds& bounds, 
		MediumProperties* mp) :
    Surface( pos, rot, mp),  m_phiSpan(0.,0.),
    theBounds( bounds.clone()) { computePhiSpan();}

  BoundSurface( const BoundSurface& iToCopy) :
    Surface( iToCopy ), 
    m_phiSpan(iRHS.m_phiSpan),
    theBounds( iToCopy.theBounds->clone() ) {}

  const BoundSurface& operator=(const BoundSurface& iRHS ) {
    Surface::operator=(iRHS);
    m_phiSpan=iRHS.m_phiSpan;
    theBounds.reset( iRHS.theBounds->clone() );
    return *this;
  }

  const Bounds& bounds() const { return *theBounds;}

  std::pair<float,float> const & phiSpan() const { return m_phiSpan;}

protected:
  void computePhiSpan();

private:

  std::pair<float,float> m_phiSpan;

  //own_ptr<Bounds,OwnerPolicy::Clone> theBounds;
  std::auto_ptr<Bounds> theBounds;
};



#endif // Geom_BoundSurface_H
