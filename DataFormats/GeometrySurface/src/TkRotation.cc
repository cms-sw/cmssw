#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include <iostream>

std::ostream & operator<< <float>( std::ostream& s, const TkRotation<float>& rtmp) {
  return s << " (" << rtmp.xx() << ',' << rtmp.xy() << ',' << rtmp.xz() << ")\n"
	   << " (" << rtmp.yx() << ',' << rtmp.yy() << ',' << rtmp.yz() << ")\n"
	   << " (" << rtmp.zx() << ',' << rtmp.zy() << ',' << rtmp.zz() << ") ";
} 

std::ostream & operator<< <double>( std::ostream& s, const TkRotation<double>& rtmp) {
  return s << " (" << rtmp.xx() << ',' << rtmp.xy() << ',' << rtmp.xz() << ")\n"
	   << " (" << rtmp.yx() << ',' << rtmp.yy() << ',' << rtmp.yz() << ")\n"
	   << " (" << rtmp.zx() << ',' << rtmp.zy() << ',' << rtmp.zz() << ") ";
} 


namespace geometryDetails {
  void TkRotationErr1() {
    std::cerr << "TkRotation: zero axis" << std::endl;
  }
  void TkRotationErr2() {
    std::cerr << "TkRotation::rotateAxes: bad axis vectors" << std::endl;
  }

}
