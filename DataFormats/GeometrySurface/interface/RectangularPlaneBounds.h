#ifndef Geom_RectangularPlaneBounds_H
#define Geom_RectangularPlaneBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <cmath>

/** \class RectangularPlaneBounds
 *  Rectangular plane bounds.
 *  Local Coordinate system coincides with center of the box
 *  with X axis along the width and Y axis along the lenght.
 */
class RectangularPlaneBounds : public Bounds {
public:

  /// Construct from  half width (extension in local X),
  /// half length (Y) and half thickness (Z)
  RectangularPlaneBounds( float w, float h, float t) : 
    halfWidth(w), halfLength(h), halfThickness(t) {}

  /// Lenght along local Y
  virtual float length()    const { return 2*halfLength;}
  /// Width along local X
  virtual float width()     const { return 2*halfWidth;}
  /// Thickness of the volume in local Z 
  virtual float thickness() const { return 2*halfThickness;}

  // basic bounds function
  virtual bool inside( const Local2DPoint& p) const {
    return fabs(p.x()) <= halfWidth && fabs(p.y()) <= halfLength;
  }

  virtual bool inside( const Local3DPoint& p) const {
    return fabs(p.x()) <= halfWidth && 
           fabs(p.y()) <= halfLength &&
           fabs(p.z()) <= halfThickness;
  }

  virtual bool inside( const Local3DPoint& p, const LocalError& err,
		       float scale=1.) const {
    //   RectangularPlaneBounds tmp( halfWidth +  std::sqrt(err.xx())*scale,
    //				halfLength + std::sqrt(err.yy())*scale,
    //				halfThickness);
    //     return tmp.inside(p);

    return 
      fabs(p.z()) <= halfThickness &&
      (fabs(p.x()) <= halfWidth  || fabs(p.x()) <= halfWidth  + std::sqrt(err.xx())*scale) &&
      (fabs(p.y()) <= halfLength || fabs(p.y()) <= halfLength + std::sqrt(err.yy())*scale);
  }
    
  virtual bool inside( const Local2DPoint& p, const LocalError& err, float scale=1.) const {
    //    return Bounds::inside(p,err,scale);
    return 
      (fabs(p.x()) <= halfWidth  || fabs(p.x()) <= halfWidth  + std::sqrt(err.xx())*scale) &&
      (fabs(p.y()) <= halfLength || fabs(p.y()) <= halfLength + std::sqrt(err.yy())*scale);
  }

  virtual Bounds* clone() const { 
    return new RectangularPlaneBounds(*this);
  }


private:
  float halfWidth;
  float halfLength;
  float halfThickness;

};

#endif
