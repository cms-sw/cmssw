
#include "DataFormats/GeometrySurface/interface/GeneralTrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include <cmath>

GeneralTrapezoidalPlaneBounds::GeneralTrapezoidalPlaneBounds( float be, float te, float ang, 
							      float a, float t) : 
  hbotedge(be), htopedge(te), tilt_angle(ang), hapothem(a), hthickness(t) {

  // pre-compute offsets of triangles and tg of (half) opening
  // angles of the trapezoid for faster inside() implementation
    
  tg_tilt = tan(ang*M_PI/180.);
  xoff = a * tg_tilt;
  offsetp = a * (te+be) / (te-be+2.*xoff);  // for x > 0 
  tan_ap = (te+xoff) / (offsetp + a);
  offsetn = a * (te+be) / (te-be-2.*xoff);  // for x < 0 
  tan_an = (xoff-te) / (offsetn + a);
}

bool GeneralTrapezoidalPlaneBounds::inside( const Local2DPoint& p) const {
  return std::abs(p.y()) <= hapothem && 
    ( (p.x() >=0. && p.x()/(p.y()+offsetp) <= tan_ap) ||
      (p.x() < 0. && p.x()/(p.y()+offsetn) >= tan_an) );
}

bool GeneralTrapezoidalPlaneBounds::inside( const Local3DPoint& p) const {
  return std::abs(p.y()) <= hapothem &&
    ( (p.x() >=0. && p.x()/(p.y()+offsetp) <= tan_ap) ||
      (p.x() < 0. && p.x()/(p.y()+offsetn) >= tan_an) ) &&
    std::abs(p.z()) <= hthickness;
}

bool GeneralTrapezoidalPlaneBounds::inside( const Local3DPoint& p, 
					    const LocalError& err, float scale) const {

  GeneralTrapezoidalPlaneBounds tmp( hbotedge + sqrt(err.xx())*scale,
				     htopedge + sqrt(err.xx())*scale,
				     tilt_angle,
				     hapothem + sqrt(err.yy())*scale,
				     hthickness);
  return tmp.inside(p);
}

const std::vector<float> GeneralTrapezoidalPlaneBounds::parameters() const { 
  std::vector<float> vec(7);
  // Same order as geant3 for constructor compatibility
  vec[0] = hthickness;
  vec[1] = 0;
  vec[2] = 0;
  vec[3] = hapothem;
  vec[4] = hbotedge;
  vec[5] = htopedge;
  vec[6] = tilt_angle;
  return vec;
}
