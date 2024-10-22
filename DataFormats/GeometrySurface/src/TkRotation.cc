#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include <iostream>

template <>
std::ostream& operator<< <float>(std::ostream& s, const TkRotation<float>& rtmp) {
  return s << " (" << rtmp.xx() << ',' << rtmp.xy() << ',' << rtmp.xz() << ")\n"
           << " (" << rtmp.yx() << ',' << rtmp.yy() << ',' << rtmp.yz() << ")\n"
           << " (" << rtmp.zx() << ',' << rtmp.zy() << ',' << rtmp.zz() << ") ";
}

template <>
std::ostream& operator<< <double>(std::ostream& s, const TkRotation<double>& rtmp) {
  return s << " (" << rtmp.xx() << ',' << rtmp.xy() << ',' << rtmp.xz() << ")\n"
           << " (" << rtmp.yx() << ',' << rtmp.yy() << ',' << rtmp.yz() << ")\n"
           << " (" << rtmp.zx() << ',' << rtmp.zy() << ',' << rtmp.zz() << ") ";
}

template <>
std::ostream& operator<< <float>(std::ostream& s, const TkRotation2D<float>& rtmp) {
  return s << rtmp.x() << "\n" << rtmp.y();
}

template <>
std::ostream& operator<< <double>(std::ostream& s, const TkRotation2D<double>& rtmp) {
  return s << rtmp.x() << "\n" << rtmp.y();
}

namespace geometryDetails {
  void TkRotationErr1() { std::cerr << "TkRotation: zero axis" << std::endl; }
  void TkRotationErr2() { std::cerr << "TkRotation::rotateAxes: bad axis vectors" << std::endl; }

}  // namespace geometryDetails
