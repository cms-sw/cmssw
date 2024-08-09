#ifndef RecoTracker_LSTCore_interface_EndcapGeometry_h
#define RecoTracker_LSTCore_interface_EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace lst {
  class EndcapGeometry {
  private:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    std::vector<unsigned int> geoMapDetId_buf;
    std::vector<float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry() = default;
    EndcapGeometry(std::string const& filename);
    ~EndcapGeometry() = default;

    void load(std::string const&);
    void fillGeoMapArraysExplicit();
    float getdxdy_slope(unsigned int detid) const;
  };
}  // namespace lst

#endif
