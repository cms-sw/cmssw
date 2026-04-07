#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

using namespace lstgeometry;

Geometry::Geometry(Sensors sensors,
                   std::array<float, kBarrelLayers> const &average_r_barrel,
                   std::array<float, kEndcapLayers> const &average_z_endcap,
                   float pt_cut) {
  auto slopes = computeSlopes(sensors);
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto binned_detids = binDetIds(sensors);

  pixel_map = buildPixelMap(sensors, pt_cut);

  module_map = buildModuleMap(sensors, binned_detids, average_r_barrel, average_z_endcap, pt_cut);

  // Drop all the extra data that is no longer needed
  for (auto &[detId, sensor] : sensors) {
    sensor.extra.reset();
  }

  this->sensors = std::move(sensors);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(lstgeometry::Geometry);
