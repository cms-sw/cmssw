/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_PPSTimingCalibration_h
#define CondFormats_PPSObjects_PPSTimingCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <vector>

class PPSTimingCalibration {
public:
  /// Helper structure for indexing calibration data
  struct Key {
    int db, sampic, channel, cell;

    /// Comparison operator
    bool operator<(const Key& rhs) const;
    friend std::ostream& operator<<(std::ostream& os, const Key& key);

    COND_SERIALIZABLE;
  };
  //--------------------------------------------------------------------------

  using ParametersMap = std::map<Key, std::vector<double> >;
  using TimingMap = std::map<Key, std::pair<double, double> >;

  PPSTimingCalibration() = default;
  PPSTimingCalibration(const std::string& formula, const ParametersMap& params, const TimingMap& timeinfo)
      : formula_(formula), parameters_(params), timeInfo_(timeinfo) {}
  ~PPSTimingCalibration() = default;

  std::vector<double> parameters(int key1, int key2, int key3, int key4) const;
  inline const std::string& formula() const { return formula_; }
  double timeOffset(int key1, int key2, int key3, int key4 = -1) const;
  double timePrecision(int key1, int key2, int key3, int key4 = -1) const;

  friend std::ostream& operator<<(std::ostream& os, const PPSTimingCalibration& data);

private:
  std::string formula_;
  ParametersMap parameters_;
  TimingMap timeInfo_;

  COND_SERIALIZABLE;
};

#endif
