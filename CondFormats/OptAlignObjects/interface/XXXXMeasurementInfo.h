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
  std::string objectName_;
  std::string objectType_;
  std::vector<std::string> measObjectNames_;
  std::vector<OpticalAlignParam> values_;
  unsigned long objectID_;

/*   void clear() { */
/*     objectName_ = ""; */
/*     objectType_ = ""; */
/*     measObjectNames_.clear(); */
/*     objectID_ = 0; */
/*     values_.clear(); */
/*   } */
};

#endif //XXXXMeasureInfo_H
