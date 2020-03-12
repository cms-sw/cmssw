#ifndef OpticalAlignMeasurements_H
#define OpticalAlignMeasurements_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"

#include <vector>
#include <iostream>

/**
  easy output...
**/

class OpticalAlignMeasurements;

std::ostream &operator<<(std::ostream &, const OpticalAlignMeasurements &);

/**
   Description: Class for OpticalAlignMeasurements for use by COCOA.
 **/
class OpticalAlignMeasurements {
public:
  OpticalAlignMeasurements() {}
  virtual ~OpticalAlignMeasurements() {}

  std::vector<OpticalAlignMeasurementInfo> oaMeasurements_;

  COND_SERIALIZABLE;
};

#endif  // OpticalAlignMeasurements_H
