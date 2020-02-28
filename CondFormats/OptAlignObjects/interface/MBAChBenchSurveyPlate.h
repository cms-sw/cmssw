#ifndef MBAChBenchSurveyPlate_H
#define MBAChBenchSurveyPlate_H

#include "CondFormats/Serialization/interface/Serializable.h"

/* #include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h" */

#include <vector>
#include <iostream>
#include <string>

/**
  easy output...
**/

/* class MBAChBenchSurveyPlate; */

class MBAChBenchSurveyPlateData {
public:
  int edmsID_;
  int surveyCode_;
  int line_;
  int plate_;
  std::string side_;
  int object_;
  float posX_;
  float posY_;
  float posZ_;
  long long measDateTime_;

  COND_SERIALIZABLE;
};

/**
   Description: Class for MBAChBenchSurveyPlate for use as calibration.
 **/
class MBAChBenchSurveyPlate {
public:
  MBAChBenchSurveyPlate() {}
  virtual ~MBAChBenchSurveyPlate() {}
  std::vector<MBAChBenchSurveyPlateData> mbaChBenchSurveyPlate_;

  COND_SERIALIZABLE;
};

#endif  // MBAChBenchSurveyPlate_H
