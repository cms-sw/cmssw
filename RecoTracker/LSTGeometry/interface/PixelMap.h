#ifndef RecoTracker_LSTGeometry_interface_PixelMap_h
#define RecoTracker_LSTGeometry_interface_PixelMap_h

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"

namespace lstgeometry {

  using LayerSubdetChargeKey = std::tuple<unsigned int, unsigned int, int>;
  using LayerSubdetChargeMap = std::unordered_map<LayerSubdetChargeKey,
                                                  std::vector<std::unordered_set<unsigned int>>,
                                                  boost::hash<LayerSubdetChargeKey>>;
  using PixelMap = LayerSubdetChargeMap;

  PixelMap buildPixelMap(Sensors const& sensors, float pt_cut);

}  // namespace lstgeometry

#endif
