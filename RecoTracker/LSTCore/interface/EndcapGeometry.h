#ifndef RecoTracker_LSTCore_interface_EndcapGeometry_h
#define RecoTracker_LSTCore_interface_EndcapGeometry_h

#include <map>
#include <string>
#include <vector>

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

    void load(std::string const&);
    void fillGeoMapArraysExplicit();
    float getdxdy_slope(unsigned int detid) const;
  };
}  // namespace lst

#endif
