/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include <ostream>

//--------------------------------------------------------------------------

bool PPSTimingCalibration::Key::operator<(const PPSTimingCalibration::Key& rhs) const {
  if (db == rhs.db) {
    if (sampic == rhs.sampic) {
      if (channel == rhs.channel)
        return cell < rhs.cell;
      return channel < rhs.channel;
    }
    return sampic < rhs.sampic;
  }
  return db < rhs.db;
}

std::ostream& operator<<(std::ostream& os, const PPSTimingCalibration::Key& key) {
  return os << key.db << " " << key.sampic << " " << key.channel << " " << key.cell;
}

//--------------------------------------------------------------------------

std::vector<double> PPSTimingCalibration::parameters(int key1, int key2, int key3, int key4) const {
  Key key{key1, key2, key3, key4};
  auto out = parameters_.find(key);
  if (out == parameters_.end())
    return {};
  return out->second;
}

double PPSTimingCalibration::timeOffset(int key1, int key2, int key3, int key4) const {
  Key key{key1, key2, key3, key4};
  auto out = timeInfo_.find(key);
  if (out == timeInfo_.end())
    return 0.;
  return out->second.first;
}

double PPSTimingCalibration::timePrecision(int key1, int key2, int key3, int key4) const {
  Key key{key1, key2, key3, key4};
  auto out = timeInfo_.find(key);
  if (out == timeInfo_.end())
    return 0.;
  return out->second.second;
}

std::ostream& operator<<(std::ostream& os, const PPSTimingCalibration& data) {
  os << "FORMULA: " << data.formula_ << "\nDB SAMPIC CHANNEL CELL PARAMETERS TIME_OFFSET\n";
  for (const auto& kv : data.parameters_) {
    os << kv.first << " [";
    for (size_t i = 0; i < kv.second.size(); ++i)
      os << (i > 0 ? ", " : "") << kv.second.at(i);

    PPSTimingCalibration::Key key = kv.first;
    if (data.timeInfo_.find(key) == data.timeInfo_.end())
      key = {kv.first.db, kv.first.sampic, kv.first.channel, -1};

    const auto& time = data.timeInfo_.at(key);
    os << "] " << time.first << " " << time.second << "\n";
  }
  return os;
}
