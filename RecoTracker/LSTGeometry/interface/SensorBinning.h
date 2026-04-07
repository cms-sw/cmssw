#ifndef RecoTracker_LSTGeometry_interface_SensorBinning_h
#define RecoTracker_LSTGeometry_interface_SensorBinning_h

#include <algorithm>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  using LocationLayerThetaBinPhiBinKey = std::tuple<Location, unsigned int, unsigned int, unsigned int>;

  using BinnedDetIds = std::unordered_map<LocationLayerThetaBinPhiBinKey,
                                          std::vector<unsigned int>,
                                          boost::hash<LocationLayerThetaBinPhiBinKey>>;

  // We split modules into overlapping theta-phi bins so that it's easier to construct module maps
  static constexpr unsigned int kNThetaBins = 4;
  static constexpr float kThetaBinRad = std::numbers::pi_v<float> / kNThetaBins;
  static constexpr unsigned int kNPhiBins = 6;
  static constexpr float kPhiBinWidth = 2 * std::numbers::pi_v<float> / kNPhiBins;

  bool isInThetaPhiBin(float theta, float phi, unsigned int theta_bin, unsigned int phi_bin);
  std::pair<unsigned int, unsigned int> getThetaPhiBins(float theta, float phi);
  std::pair<float, float> getCompatibleEtaRange(Sensor const& sensor, float zmin_bound, float zmax_bound);
  std::pair<std::pair<float, float>, std::pair<float, float>> getCompatiblePhiRange(Sensor const& sensor,
                                                                                    float ptmin,
                                                                                    float ptmax);
  BinnedDetIds binDetIds(Sensors const& sensors);

}  // namespace lstgeometry

#endif
