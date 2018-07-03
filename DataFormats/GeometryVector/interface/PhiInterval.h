#ifndef GeometryVector_PhiInterval_H
#define GeometryVector_PhiInterval_H
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

class PhiInterval {
public:
  PhiInterval(float phi1, float phi2) {
    phi2 = proxim(phi2,phi1);
    constexpr float c1 = 2.*M_PI;
    if (phi2<phi1) phi2+=c1;
    auto dphi = 0.5f*(phi2-phi1);
    auto phi = phi1+dphi;
    x = std::cos(phi);
    y = std::sin(phi);
    dcos = std::cos(dphi);
  }

  PhiInterval(float ix, float iy, float dphi) {
    auto norm = 1.f/std::sqrt(ix*ix+iy*iy);
    x = ix*norm;
    y = iy*norm;
    dcos = std::cos(dphi);
  }

  template<typename T>
  bool inside(Basic3DVector<T> const & v) const {
    return inside(v.x(),v.y());
  }

  bool inside(float ix, float iy) const {
    return ix*x+iy*y > dcos*std::sqrt(ix*ix+iy*iy);
  }

private:
  float x,y;
  float dcos;

};

#endif

