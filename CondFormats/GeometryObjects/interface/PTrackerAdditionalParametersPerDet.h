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

  enum IntParamIndex { GEOGRAPHICAL_ID = 0, BIGPIXELS_X = 1, BIGPIXELS_Y = 2, ISIZE = 3 };
  enum BoolParamIndex { BSIZE = 0 };
  enum FloatParamIndex { BIGPIXELS_PITCH_X = 0, BIGPIXELS_PITCH_Y = 1, FSIZE = 2 };

  int getGeographicalId(int theIndex) const;
  int bigPixelsX(int theIndex) const;
  int bigPixelsY(int theIndex) const;
  float bigPixelsPitchX(int theIndex) const;
  float bigPixelsPitchY(int theIndex) const;

  std::vector<int> getAllGeographicalIds() const;
  std::vector<int> allBigPixelsXs() const;
  std::vector<int> allBigPixelsYs() const;
  std::vector<float> allBigPixelsPitchXs() const;
  std::vector<float> allBigPixelsPitchYs() const;

  void setGeographicalId(int geographicalId);
  void setBigPixelsX(int bigPixelsX);
  void setBigPixelsY(int bigPixelsY);
  void setBigPixelsPitchX(float bigPixelsPitchX);
  void setBigPixelsPitchY(float bigPixelsPitchY);

  std::vector<std::vector<int>> intParams_;
  std::vector<std::vector<bool>> boolParams_;
  std::vector<std::vector<float>> floatParams_;

  COND_SERIALIZABLE;
};

#endif
