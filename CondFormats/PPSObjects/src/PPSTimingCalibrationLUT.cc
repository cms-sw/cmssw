/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Christopher Misan
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"
#include <ostream>

//--------------------------------------------------------------------------

bool PPSTimingCalibrationLUT::Key::operator<(const PPSTimingCalibrationLUT::Key& rhs) const {
  if (sector == rhs.sector) {
    if (station == rhs.station) {
      if (plane == rhs.plane)
        return channel < rhs.channel;
      return plane < rhs.plane;
    }
    return station < rhs.station;
  }
  return sector < rhs.sector;
}

std::ostream& operator<<(std::ostream& os, const PPSTimingCalibrationLUT::Key& key) {
  return os << key.sector << " " << key.station << " " << key.plane << " " << key.channel;
}

//--------------------------------------------------------------------------

std::vector<double> PPSTimingCalibrationLUT::bins(int key1, int key2, int key3, int key4) const {
  Key key{key1, key2, key3, key4};
  auto out = binMap_.find(key);
  if (out == binMap_.end())
    return {};
  return out->second;
}

std::ostream& operator<<(std::ostream& os, const PPSTimingCalibrationLUT& data) {
  os << "\nSECTOR STATION PLANE CHANNEL SAMPLES \n";
  for (const auto& kv : data.binMap_) {
    os << kv.first << "\n[";
    for (size_t i = 0; i < kv.second.size(); ++i)
      os << (i > 0 ? ", " : "") << kv.second.at(i);
    os << " ]\n";
  }
  return os;
}
