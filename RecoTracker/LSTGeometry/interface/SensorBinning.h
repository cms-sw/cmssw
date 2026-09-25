#ifndef RecoTracker_LSTGeometry_interface_SensorBinning_h
#define RecoTracker_LSTGeometry_interface_SensorBinning_h

#include <algorithm>
#include <array>
#include <vector>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  // We split modules into overlapping theta-phi bins so that it's easier to construct module maps
  inline constexpr unsigned int kNThetaBins = 4;
  inline constexpr float kThetaBinRad = std::numbers::pi_v<float> / kNThetaBins;
  inline constexpr unsigned int kNPhiBins = 6;
  inline constexpr float kPhiBinWidth = 2 * std::numbers::pi_v<float> / kNPhiBins;

  using BinnedDetIds =
      std::array<std::array<std::array<std::array<std::vector<unsigned int>, kNPhiBins>, kNThetaBins>, kBarrelLayers + 1>,
                 2>;

  inline unsigned int locationIndex(Location location) { return location == Location::barrel ? 0 : 1; }

  inline std::vector<unsigned int>& binnedDetIdsAt(
      BinnedDetIds& binned_detids, Location location, unsigned int layer, unsigned int thetaBin, unsigned int phiBin) {
    return binned_detids[locationIndex(location)][layer][thetaBin][phiBin];
  }

  inline std::vector<unsigned int> const& binnedDetIdsAt(BinnedDetIds const& binned_detids,
                                                         Location location,
                                                         unsigned int layer,
                                                         unsigned int thetaBin,
                                                         unsigned int phiBin) {
    return binned_detids[locationIndex(location)][layer][thetaBin][phiBin];
  }

  bool isInThetaPhiBin(float theta, float phi, unsigned int thetaBin, unsigned int phiBin);
  std::pair<unsigned int, unsigned int> getThetaPhiBins(float theta, float phi);
  std::pair<float, float> getCompatibleEtaRange(Sensor const& sensor, float zmin_bound, float zmax_bound);
  std::pair<std::pair<float, float>, std::pair<float, float>> getCompatiblePhiRange(Sensor const& sensor,
                                                                                    float ptmin,
                                                                                    float ptmax);
  BinnedDetIds binDetIds(Sensors const& sensors);

}  // namespace lstgeometry

#endif
