#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h" 
#include <cmath>

RectangularPlaneBounds::RectangularPlaneBounds( float w, float h, float t) : 
  halfWidth(w), halfLength(h), halfThickness(t) {}


RectangularPlaneBounds::~RectangularPlaneBounds(){}

bool RectangularPlaneBounds::inside(const Local3DPoint& p, const LocalError& err,
				    float scale) const {
  if( (scale >=0) && inside(p) ) return true;
  return
      std::abs(p.z()) <  halfThickness &&
      (std::abs(p.x()) < halfWidth  + std::sqrt(err.xx())*scale) &&
      (std::abs(p.y()) < halfLength + std::sqrt(err.yy())*scale);
}
    
bool RectangularPlaneBounds::inside( const Local2DPoint& p, const LocalError& err, 
				     float scale) const {
  if( (scale >=0) && inside(p) ) return true;
  return 
      (std::abs(p.x()) < halfWidth  + std::sqrt(err.xx())*scale) &&
      (std::abs(p.y()) < halfLength + std::sqrt(err.yy())*scale);
}

float RectangularPlaneBounds::significanceInside(const Local3DPoint& p, const LocalError& err) const {
   return std::max((std::abs(p.x()) - halfWidth )/std::sqrt(err.xx()),
                   (std::abs(p.y()) - halfLength)/std::sqrt(err.yy())
                  );
}


std::pair<bool,bool> RectangularPlaneBounds::inout( const Local3DPoint& p, const LocalError& err, float scale) const {
  float xl = std::abs(p.x()) -  std::sqrt(err.xx())*scale;
  float xh = std::abs(p.x()) +  std::sqrt(err.xx())*scale;
  bool  inx = xl<halfWidth;
  bool outx = xh>halfWidth;
  
  float yl = std::abs(p.y()) -  std::sqrt(err.yy())*scale;
  float yh = std::abs(p.y()) +  std::sqrt(err.yy())*scale;
  bool  iny = yl<halfLength;
  bool outy = yh>halfLength;

  return std::pair<bool,bool>(inx&&iny,outx||outy);

}

Bounds* RectangularPlaneBounds::clone() const { 
  return new RectangularPlaneBounds(*this);
}


