#ifndef RecoTracker_LSTGeometry_interface_Geometry_h
#define RecoTracker_LSTGeometry_interface_Geometry_h

#include <array>
#include <optional>

#include "RecoTracker/LSTGeometry/interface/ModuleMap.h"
#include "RecoTracker/LSTGeometry/interface/PixelMap.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

namespace lstgeometry {

  struct Geometry {
    Sensors sensors;
    Slopes barrel_slopes;
    Slopes endcap_slopes;
    PixelMap pixel_map;
    ModuleMap module_map;

    Geometry(Sensors sensors,
             std::array<float, kBarrelLayers> const &average_r_barrel,
             std::array<float, kEndcapLayers> const &average_z_endcap,
             float ptCut);
  };

}  // namespace lstgeometry

#endif
