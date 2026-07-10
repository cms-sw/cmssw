#ifndef RecoTracker_LSTGeometry_interface_PixelMap_h
#define RecoTracker_LSTGeometry_interface_PixelMap_h

#include <functional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"

namespace lstgeometry {

  using LayerSubdetChargeKey = std::tuple<unsigned int, unsigned int, int>;
  struct LayerSubdetChargeHash {
    std::size_t operator()(LayerSubdetChargeKey const& key) const {
      auto [layer, subdet, charge] = key;
      std::size_t seed = std::hash<unsigned int>{}(layer);
      seed ^= std::hash<unsigned int>{}(subdet) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>{}(charge) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };
  using LayerSubdetChargeMap =
      std::unordered_map<LayerSubdetChargeKey, std::vector<std::vector<unsigned int>>, LayerSubdetChargeHash>;
  using PixelMap = LayerSubdetChargeMap;

  PixelMap buildPixelMap(Sensors const& sensors, float pt_cut);

}  // namespace lstgeometry

#endif
