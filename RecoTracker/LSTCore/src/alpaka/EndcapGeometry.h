#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace SDL {

  // FIXME: Need to separate this better into host and device classes
  // This is only needed for host, but we template it to avoid symbol conflicts
  template <typename TDev>
  class EndcapGeometryHost;

  template <>
  class EndcapGeometryHost<Dev> {
  public:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

    EndcapGeometryHost() = default;
    ~EndcapGeometryHost() = default;

    void load(std::string);
    float getdxdy_slope(unsigned int detid) const;
  };

  template <typename TDev>
  class EndcapGeometry;

  template <>
  class EndcapGeometry<Dev> {
  private:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    Buf<SDL::Dev, unsigned int> geoMapDetId_buf;
    Buf<SDL::Dev, float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry(Dev const& devAccIn, QueueAcc& queue, SDL::EndcapGeometryHost<Dev> const& endcapGeometryIn);
    ~EndcapGeometry() = default;

    void fillGeoMapArraysExplicit(QueueAcc& queue);
    float getdxdy_slope(unsigned int detid) const;
  };
}  // namespace SDL

#endif
