#ifndef CondFormats_PCLConfig_AlignPCLThresholdsHG_h
#define CondFormats_PCLConfig_AlignPCLThresholdsHG_h

#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>
#include <vector>

class AlignPCLThresholdsHG : public AlignPCLThresholds {
public:
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

  void SetFractionCut(const std::string &AlignableId, const coordType &type, const float &cut);

  const std::unordered_map<std::string, std::vector<float>> &getFloatMap() const { return floatMap; }
  const std::vector<float> &getFloatVec(const std::string &AlignableId) const;

  float getFractionCut(const std::string &AlignableId, const coordType &type) const;
  std::array<float, 6> getFractionCut(const std::string &AlignableId) const;

  int payloadVersion() const;

  void printAllHG() const;

  ~AlignPCLThresholdsHG() override {}

private:
  std::unordered_map<std::string, std::vector<float>> floatMap;
  std::unordered_map<std::string, std::vector<int>> intMap;
  std::unordered_map<std::string, std::vector<std::string>> stringMap;

  COND_SERIALIZABLE;
};

#endif
