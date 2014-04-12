#ifndef GeneralNSurfaceDelimitedBounds_H
#define GeneralNSurfaceDelimitedBounds_H

#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include <vector>

/** Bounds for a surface of any type, delimited by N other surfaces of any type.
 *  A point is "inside" if it is on the correct side of all delimiting surfaces.
 *  This way of computing "inside" is relatively expensive, and should only be applied
 *  to situations where there is no specialized implementation for the concrete
 *  surface types. 
 */

class GeneralNSurfaceDelimitedBounds GCC11_FINAL : public Bounds {
public:

    typedef std::pair<const Surface*, SurfaceOrientation::Side>  SurfaceAndSide;
    typedef std::vector<SurfaceAndSide>                          SurfaceContainer;

    GeneralNSurfaceDelimitedBounds( const Surface*  surf, 
				    const std::vector<SurfaceAndSide>& limits) :
	theLimits( limits), theSurface(surf) {}

  virtual float length()    const { return 0;}
  virtual float width()     const { return 0;}
  virtual float thickness() const { return 0;}


  virtual bool inside( const Local3DPoint& lp) const {
    return myInside(lp,0);
  }
    
  virtual bool inside( const Local3DPoint&, const LocalError&, float scale=1.f) const;

  virtual Bounds* clone() const {return new GeneralNSurfaceDelimitedBounds(*this);}
    
private:

    SurfaceContainer theLimits;
    const Surface*   theSurface;

    bool myInside( const Local3DPoint& lp, float tolerance) const;

};

#endif
