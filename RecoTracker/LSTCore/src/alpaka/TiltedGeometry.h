#ifndef TiltedGeometry_h
#define TiltedGeometry_h

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

namespace SDL {
  template <typename>
  class TiltedGeometry;
  template <>
  class TiltedGeometry<SDL::Dev> {
  private:
    std::map<unsigned int, float> drdzs_;  // dr/dz slope
    std::map<unsigned int, float> dxdys_;  // dx/dy slope

  public:
    TiltedGeometry() = default;
    TiltedGeometry(std::string filename);
    ~TiltedGeometry() = default;

    void load(std::string);

    float getDrDz(unsigned int detid) const;
    float getDxDy(unsigned int detid) const;
  };

}  // namespace SDL

#endif
