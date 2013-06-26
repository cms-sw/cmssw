#ifndef MBAChBenchCalPlate_H
#define MBAChBenchCalPlate_H

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
};

/**
   Description: Class for MBAChBenchCalPlate for use as calibration.
 **/
class MBAChBenchCalPlate {
 public:
  MBAChBenchCalPlate() {}
  virtual ~MBAChBenchCalPlate() {}
  std::vector<MBAChBenchCalPlateData> mbaChBenchCalPlate_;
};

#endif // MBAChBenchCalPlate_H
