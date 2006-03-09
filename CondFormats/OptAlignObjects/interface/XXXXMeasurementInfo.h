#ifndef XXXXMeasurementInfo_H
#define XXXXMeasurementInfo_H

#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

/**
  easy output...
**/

class XXXXMeasurementInfo;

std::ostream & operator<<(std::ostream &, const XXXXMeasurementInfo &);

// a Class holding data for an Optical Alignment Measurement
/**
    Author:  Michael Case
    Date:    March 7, 2006

 **/

class  XXXXMeasurementInfo {
 public:  
  OpticalAlignParam x1_, x2_, x3_, x4_;
  std::vector<OpticalAlignParam> extraEntries_;
  std::string objectType_;
  unsigned long objectID_;
  void clear() {
    x1_.clear();
    x2_.clear();
    x3_.clear();
    x4_.clear();
    extraEntries_.clear();
    objectType_.clear();
    objectID_ = 0;
  }
};

#endif //XXXXMeasureInfo_H
