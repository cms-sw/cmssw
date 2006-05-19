#include "CondFormats/OptAlignObjects/interface/XXXXMeasurementInfo.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream & os, const XXXXMeasurementInfo & r)
{
  os << "Name: " << r.objectName_ << " Type: " << r.objectType_ << "  ID: " << r.objectID_ << std::endl;
  int iw = os.width(); // save current width
  int ip = os.precision(); // save current precision
  int now = 12;
  //  int nop = 5;

  std::vector<std::string>::const_iterator item;

  for(item = r.measObjectNames_.begin(); item != r.measObjectNames_.end(); ++item ){
    os << std::setw( now ) << *item << std::endl;
  }

  std::vector<OpticalAlignParam>::const_iterator iteo;
  for( iteo = r.values_.begin(); iteo != r.values_.end(); ++iteo ){
    os << *iteo;
  }

  os << std::setprecision( ip ) << std::setw( iw );
  return os;
}


