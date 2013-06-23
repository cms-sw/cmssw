#ifndef APPROX_ERF_H
#define APPROX_ERF_H
#include "DataFormats/Math/interface/approx_exp.h"

inline
float approx_erf(float x) {
  auto xx = std::min(std::abs(x),5.f);
  xx*=xx;
  return std::copysign(std::sqrt(1.f- unsafe_expf<4>(-xx*(1.f+0.2733f/(1.f+0.147f*xx)) )),x);
  // return std::sqrt(1.f- std::exp(-x*x*(1.f+0.2733f/(1.f+0.147f*x*x)) ));
}


#endif
