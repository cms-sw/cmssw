#ifndef Geom_GeneralTrapezoidalPlaneBounds_H
#define Geom_GeneralTrapezoidalPlaneBounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <vector>

/**
 * GeneralTrapezoidal detector bounds.
 * Local Coordinate system coincides with center of the box
 * with y axis being the symmetry axis along the height
 * and pointing in the direction of top_edge.
 */

class GeneralTrapezoidalPlaneBounds : public Bounds {
public:

  GeneralTrapezoidalPlaneBounds( float be, float te, float ang, 
				 float a, float t);
  
  virtual float length() const    { return 2 * hapothem;}
  virtual float width()  const    { return 2 * std::max( hbotedge, htopedge);}
  virtual float thickness() const { return 2 * hthickness;}

  virtual float widthAtHalfLength() const {return hbotedge+htopedge;}

  virtual bool inside( const Local2DPoint& p) const;

  virtual bool inside( const Local3DPoint& p) const;

  virtual bool inside( const Local3DPoint& p, const LocalError& err, float scale) const;

  virtual bool inside( const Local2DPoint& p, const LocalError& err, float scale) const {
    return Bounds::inside(p,err,scale);
  }

  virtual const std::vector<float> parameters() const;

  virtual Bounds* clone() const { 
    return new GeneralTrapezoidalPlaneBounds(*this);
  }

private:
  // persistent part
  float hbotedge;
  float htopedge;
  float tilt_angle;
  float hapothem;
  float hthickness;

  // transient part 
  float xoff;
  float tg_tilt;
  float offsetp,offsetn;
  float tan_ap,tan_an;
};

#endif
