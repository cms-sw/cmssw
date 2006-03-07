#ifndef XXXXMeasurements_H
#define XXXXMeasurements_H

#include "CondFormats/OptAlignObjects/interface/XXXXMeasurementInfo.h"

#include <vector>
#include <iostream>

/**
  easy output...
**/

class XXXXMeasurements;

std::ostream & operator<<(std::ostream &, const XXXXMeasurements &);

/**
   Description: Class for XXXXMeasurements for use by COCOA.
 **/
class XXXXMeasurements {
public:
  XXXXMeasurements() {}
  virtual ~XXXXMeasurements() {}

  std::vector<XXXXMeasurementInfo> xxxxMeasurements_;
};

#endif // XXXXMeasurements_H
