#ifndef RecoTracker_LSTGeometry_interface_ModuleMapMethods_h
#define RecoTracker_LSTGeometry_interface_ModuleMapMethods_h

#include <vector>
#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"

namespace lstgeometry {

  using ModuleMap = std::unordered_map<unsigned int, std::vector<unsigned int>>;

  ModuleMap buildModuleMap(Sensors const& sensors,
                           BinnedDetIds const& binned_detids,
                           std::array<float, kBarrelLayers> const& average_r_barrel,
                           std::array<float, kEndcapLayers> const& average_z_endcap,
                           float pt_cut);

}  // namespace lstgeometry

#endif
