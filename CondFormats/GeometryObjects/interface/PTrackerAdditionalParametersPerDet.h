#ifndef CondFormats_PTrackerAdditionalParametersPerDet_h
#define CondFormats_PTrackerAdditionalParametersPerDet_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>

class PTrackerAdditionalParametersPerDet {
public:
  PTrackerAdditionalParametersPerDet() {
    intParams_.resize(ISIZE, std::vector<int>(0, 0));
    floatParams_.resize(FSIZE, std::vector<float>(0, 0.));
    boolParams_.resize(BSIZE, std::vector<bool>(0, false));
  };
  ~PTrackerAdditionalParametersPerDet(){};

  enum IntParamIndex { GEOGRAPHICAL_ID = 0, ISIZE = 1 };
  enum BoolParamIndex { BRICKEDNESS = 0, BSIZE = 1 };
  enum FloatParamIndex { FSIZE = 0 };

  int getGeographicalId(int theIndex) const;
  bool getBricked(int theIndex) const;
  std::vector<int> getAllGeographicalIds() const;
  std::vector<bool> getAllBricked() const;

  void setGeographicalId(int geographicalId);
  void setBricked(bool isBricked);

  std::vector<std::vector<int>> intParams_;
  std::vector<std::vector<bool>> boolParams_;
  std::vector<std::vector<float>> floatParams_;

  COND_SERIALIZABLE;
};

#endif
