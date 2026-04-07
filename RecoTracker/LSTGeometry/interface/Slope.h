#ifndef RecoTracker_LSTGeometry_interface_Slope_h
#define RecoTracker_LSTGeometry_interface_Slope_h

#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  struct Slope {
    float drdz;
    float dxdy;

    Slope() = default;
    Slope(float dx, float dy, float dz);
  };

  using Slopes = std::unordered_map<unsigned int, Slope>;

  std::tuple<Slopes, Slopes> computeSlopes(Sensors const& sensors);

}  // namespace lstgeometry

#endif
