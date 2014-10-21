#ifndef Geom_RectangularPlaneBounds_H
#define Geom_RectangularPlaneBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"


/** \class RectangularPlaneBounds
 *  Rectangular plane bounds.
 *  Local Coordinate system coincides with center of the box
 *  with X axis along the width and Y axis along the lenght.
 */
class RectangularPlaneBounds GCC11_FINAL : public Bounds {
public:

  /// Construct from  half width (extension in local X),
  /// half length (Y) and half thickness (Z)
  RectangularPlaneBounds( float w, float h, float t);
  ~RectangularPlaneBounds();

  /// Lenght along local Y
  virtual float length()    const { return 2*halfLength;}
  /// Width along local X
  virtual float width()     const { return 2*halfWidth;}
  /// Thickness of the volume in local Z 
  virtual float thickness() const { return 2*halfThickness;}

  // basic bounds function
  using Bounds::inside;

  virtual bool inside( const Local2DPoint& p) const {
    return
     (std::abs(p.x()) < halfWidth) &
     (std::abs(p.y()) < halfLength);
  }

  virtual bool inside( const Local3DPoint& p) const {
    return
     (std::abs(p.x()) < halfWidth) &
     (std::abs(p.y()) < halfLength) &
     (std::abs(p.z()) < halfThickness);
  }



  virtual bool inside(const Local2DPoint& p, float tollerance) const {
    return (std::abs(p.x()) < (halfWidth  + tollerance) ) &
           (std::abs(p.y()) < (halfLength + tollerance) );
  }


  virtual bool inside( const Local3DPoint& p, const LocalError& err,
		       float scale=1.f) const;

  virtual bool inside( const Local2DPoint& p, const LocalError& err, float scale=1.f) const;

  // compatible of being inside or outside...
 std::pair<bool,bool> inout( const Local3DPoint& p, const LocalError& err, float scale=1.f) const;


  virtual Bounds* clone() const;

private:
  float halfWidth;
  float halfLength;
  float halfThickness;

};

#endif
