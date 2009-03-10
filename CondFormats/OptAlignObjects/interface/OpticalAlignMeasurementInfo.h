#ifndef OpticalAlignMeasurementInfo_H
#define OpticalAlignMeasurementInfo_H

#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

/**
  easy output...
**/

class OpticalAlignMeasurementInfo;

std::ostream & operator<<(std::ostream &, const OpticalAlignMeasurementInfo &);

// a Class holding data for an Optical Alignment Measurement
/**
    Author:  Michael Case
    Date:    March 7, 2006

 **/

class  OpticalAlignMeasurementInfo {
 public:  
  std::string type_;
  std::string name_;
  std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimulatedValue_; 
  std::vector<OpticalAlignParam> values_; //names of measurement values (H:, V:, T:, ...)  Dimension of this vector gives dimension of Measurement
  unsigned int ID_;

  void clear() {
    ID_ = 0;
    type_ = "";
    name_ = "";
    measObjectNames_.clear();
    values_.clear();
    isSimulatedValue_.clear();
  }
};

#endif //OpticalAlignMeasureInfo_H
