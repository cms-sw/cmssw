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

/**
  easy output...
**/

class OpticalAlignParam;

std::ostream & operator<<(std::ostream &, const OpticalAlignParam &);


// a Class holding data for an Optical Alignment transformation
/**
    Author:  Michael Case
    Date:    Dec. 15, 2005

    It is my understanding that each optical geometrical object
    has a position in space and possible other parameters such as
    
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
