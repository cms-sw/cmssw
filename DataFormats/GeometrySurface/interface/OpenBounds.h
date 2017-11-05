#ifndef Geom_OpenBounds_H
#define Geom_OpenBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

/// Unlimited (trivial) bounds.

class OpenBounds final : public Bounds {
public:
  
  float length() const override  { return 1000000.; } 
  float width() const override { return 1000000.; }
  float thickness() const override { return 1000000.; }

  // basic bounds function

  using Bounds::inside;

  bool inside( const Local3DPoint& p ) const override { return true;}

  bool inside( const Local3DPoint& p, 
		       const LocalError& err, float scale) const override { return true;}

  bool inside( const Local2DPoint& p, 
		       const LocalError& err, float scale) const override { return true;}

  Bounds* clone() const override { return new OpenBounds();}

};


#endif // Geom_OpenBounds_H
































