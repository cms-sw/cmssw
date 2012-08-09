

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include <cmath>

TrapezoidalPlaneBounds::TrapezoidalPlaneBounds( float be, float te, 
						float a, float t) : 
  hbotedge(be), htopedge(te), hapothem(a), hthickness(t) {

  // pre-compute offset of triangle vertex and tg of (half) opening
  // angle of the trapezoid for faster inside() implementation.

  offset = a * (te+be) / (te-be);  // check sign if te < be !!! 
  tan_a = te / fabs(offset + a);
}


int TrapezoidalPlaneBounds::yAxisOrientation() const {
  int yAx = 1;
  if(hbotedge>htopedge) yAx = -1;
  return yAx;
}

bool TrapezoidalPlaneBounds::inside( const Local2DPoint& p) const {
  return fabs(p.y()) < hapothem && 
    fabs(p.x())/fabs(p.y()+offset) < tan_a;
}

bool TrapezoidalPlaneBounds::inside( const Local3DPoint& p) const {
  return fabs(p.y()) < hapothem &&
    fabs(p.x())/fabs(p.y()+offset) < tan_a &&
    fabs(p.z()) < hthickness;
}

bool TrapezoidalPlaneBounds::inside( const Local3DPoint& p,
				     const LocalError& err, float scale) const {
  TrapezoidalPlaneBounds tmp( hbotedge + sqrt(err.xx())*scale,
			      htopedge + sqrt(err.xx())*scale,
			      hapothem + sqrt(err.yy())*scale,
			      hthickness);
  return tmp.inside(p);
}
  
bool TrapezoidalPlaneBounds::inside( const Local2DPoint& p, const LocalError& err, float scale) const {
  return Bounds::inside(p,err,scale);
}

Bounds* TrapezoidalPlaneBounds::clone() const { 
  return new TrapezoidalPlaneBounds(*this);
}



const std::array<const float, 4> TrapezoidalPlaneBounds::parameters() const { 
  // Same order as geant3 for constructor compatibility
  std::array<const float, 4> vec { { hbotedge, htopedge, hthickness, hapothem } };
  return vec;
}
