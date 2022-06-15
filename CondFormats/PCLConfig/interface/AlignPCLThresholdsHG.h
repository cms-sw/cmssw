#ifndef CondFormats_PCLConfig_AlignPCLThresholdsHG_h
#define CondFormats_PCLConfig_AlignPCLThresholdsHG_h

#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>
#include <vector>

class AlignPCLThresholdsHG : public AlignPCLThresholds {
public:
  typedef std::unordered_map<std::string, std::vector<float>> param_map;
  AlignPCLThresholdsHG() {}

  enum FloatParamIndex {
    FRACTION_CUT_X = 0,
    FRACTION_CUT_Y = 1,
    FRACTION_CUT_Z = 2,
    FRACTION_CUT_TX = 3,
    FRACTION_CUT_TY = 4,
    FRACTION_CUT_TZ = 5,
    FSIZE = 6
  };

  enum IntParamIndex { ISIZE = 0 };
  enum StringParamIndex { SSIZE = 0 };

  void setFractionCut(const std::string &AlignableId, const coordType &type, const float &cut);

  const param_map &getFloatMap() const { return floatMap_; }
  const std::vector<float> &getFloatVec(const std::string &AlignableId) const;

  float getFractionCut(const std::string &AlignableId, const coordType &type) const;
  std::array<float, 6> getFractionCut(const std::string &AlignableId) const;

  const int payloadVersion() const;

  void printAll() const;

  ~AlignPCLThresholdsHG() override = default;

private:
  param_map floatMap_;
  // yet unused, but kept for possible extensions
  std::unordered_map<std::string, std::vector<int>> intMap_;
  std::unordered_map<std::string, std::vector<std::string>> stringMap_;

  COND_SERIALIZABLE;
};

#endif
