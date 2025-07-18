#ifndef RecoTracker_LSTCore_interface_TiltedGeometry_h
#define RecoTracker_LSTCore_interface_TiltedGeometry_h

#include <map>
#include <string>
#include <vector>

namespace lst {
  class TiltedGeometry {
  private:
    std::map<unsigned int, float> drdzs_;  // dr/dz slope
    std::map<unsigned int, float> dxdys_;  // dx/dy slope

  public:
    TiltedGeometry() = default;
    TiltedGeometry(std::string const& filename);

    void load(std::string const&);

    float getDrDz(unsigned int detid) const;
    float getDxDy(unsigned int detid) const;
  };

}  // namespace lst

#endif
