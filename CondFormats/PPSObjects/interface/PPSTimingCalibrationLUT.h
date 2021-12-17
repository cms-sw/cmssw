/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Christopher Misan
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_PPSTimingCalibrationLUT_h
#define CondFormats_PPSObjects_PPSTimingCalibrationLUT_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <vector>

class PPSTimingCalibrationLUT {
public:
  /// Helper structure for indexing calibration data
  struct Key {
    int sector, station, plane, channel;

    /// Comparison operator
    bool operator<(const Key& rhs) const;
    friend std::ostream& operator<<(std::ostream& os, const Key& key);

    COND_SERIALIZABLE;
  };
  //--------------------------------------------------------------------------

  using BinMap = std::map<Key, std::vector<double> >;

  PPSTimingCalibrationLUT() = default;
  PPSTimingCalibrationLUT(const BinMap& binMap) : binMap_(binMap) {}
  ~PPSTimingCalibrationLUT() = default;

  std::vector<double> bins(int key1, int key2, int key3, int key4) const;

  friend std::ostream& operator<<(std::ostream& os, const PPSTimingCalibrationLUT& data);

private:
  BinMap binMap_;

  COND_SERIALIZABLE;
};

#endif
