#ifndef GeometryVector_EtaInterval_H
#define GeometryVector_EtaInterval_H
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

class EtaInterval {
public:
  EtaInterval(float eta1, float eta2) : z1(::sinhf(eta1)), z2(::sinhf(eta2)){}

  template<typename T>
  bool inside(Basic3DVector<T> const & v) const {
    auto z = v.z();
    auto r = v.perp();
    return (z>z1*r) & (z<z2*r);
  }

private:
  float z1, z2;

};

#endif

