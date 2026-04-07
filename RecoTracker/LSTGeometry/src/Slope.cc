#include <tuple>
#include <cmath>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

namespace lstgeometry {

  Slope::Slope(float dx, float dy, float dz) {
    float dr = sqrt(dx * dx + dy * dy);
    drdz = dz != 0 ? dr / dz : kDefaultSlope;
    dxdy = dy != 0 ? -dx / dy : kDefaultSlope;
  }

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  std::tuple<Slopes, Slopes> computeSlopes(Sensors const& sensors) {
    Slopes barrel_slopes;
    Slopes endcap_slopes;

    for (auto const& [detId, sensor] : sensors) {
      float dx = roundCoordinate(sensor.extra->corners(1, 1) - sensor.extra->corners(0, 1));
      float dy = roundCoordinate(sensor.extra->corners(1, 2) - sensor.extra->corners(0, 2));
      float dz = roundCoordinate(sensor.extra->corners(1, 0) - sensor.extra->corners(0, 0));

      Slope slope(dx, dy, dz);

      auto location = sensor.extra->location;
      bool is_tilted = sensor.extra->side != Side::Center;
      bool is_strip = sensor.extra->strip;

      if (!is_strip)
        continue;

      if (location == Location::barrel and is_tilted)
        barrel_slopes[detId] = slope;
      else if (location == Location::endcap)
        endcap_slopes[detId] = slope;
    }

    return std::make_tuple(barrel_slopes, endcap_slopes);
  }
}  // namespace lstgeometry
