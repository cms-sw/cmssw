#ifndef RecoTracker_LSTGeometry_interface_IO_h
#define RecoTracker_LSTGeometry_interface_IO_h

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "RecoTracker/LSTGeometry/interface/PixelMap.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

namespace lstgeometry {

  void writeSensorCentroids(Sensors const& sensors, std::string const& base_filename, bool binary = true);
  void writeSlopes(Slopes const& slopes, Sensors const& sensors, std::string const& base_filename, bool binary = true);
  void writeModuleConnections(std::unordered_map<unsigned int, std::unordered_set<unsigned int>> const& connections,
                              std::string const& base_filename,
                              bool binary = true);
  void writePixelMaps(PixelMap const& maps, std::string const& base_filename, bool binary = true);

}  // namespace lstgeometry

#endif
