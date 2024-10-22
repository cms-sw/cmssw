#ifndef MBAChBenchCalPlate_H
#define MBAChBenchCalPlate_H

#include "CondFormats/Serialization/interface/Serializable.h"

/* #include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h" */

#include <vector>
#include <iostream>
#include <string>

/**
  easy output...
**/

/* class MBAChBenchCalPlate; */

class MBAChBenchCalPlateData {
public:
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
   Description: Class for MBAChBenchCalPlate for use as calibration.
 **/
class MBAChBenchCalPlate {
public:
  MBAChBenchCalPlate() {}
  virtual ~MBAChBenchCalPlate() {}
  std::vector<MBAChBenchCalPlateData> mbaChBenchCalPlate_;

  COND_SERIALIZABLE;
};

#endif  // MBAChBenchCalPlate_H
