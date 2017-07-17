#ifndef Geom_OpenBounds_H
#define Geom_OpenBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

/// Unlimited (trivial) bounds.

class OpenBounds final : public Bounds {
public:
  
  virtual float length() const  { return 1000000.; } 
  virtual float width() const { return 1000000.; }
  virtual float thickness() const { return 1000000.; }

  // basic bounds function

  using Bounds::inside;

  virtual bool inside( const Local3DPoint& p ) const { return true;}

  virtual bool inside( const Local3DPoint& p, 
		       const LocalError& err, float scale) const { return true;}

  virtual bool inside( const Local2DPoint& p, 
		       const LocalError& err, float scale) const { return true;}

  virtual Bounds* clone() const { return new OpenBounds();}

};


#endif // Geom_OpenBounds_H
































