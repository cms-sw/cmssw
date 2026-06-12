#ifndef RecoTracker_LSTGeometry_interface_ModuleMap_h
#define RecoTracker_LSTGeometry_interface_ModuleMap_h

#include <array>
#include <unordered_map>
#include <vector>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"

namespace lstgeometry {

  using ModuleMap = std::unordered_map<unsigned int, std::vector<unsigned int>>;

  ModuleMap buildModuleMap(Sensors const& sensors,
                           BinnedDetIds const& binned_detids,
                           std::array<float, kBarrelLayers> const& averageRBarrel,
                           std::array<float, kEndcapLayers> const& averageZEndcap,
                           float pt_cut);

}  // namespace lstgeometry

#endif
