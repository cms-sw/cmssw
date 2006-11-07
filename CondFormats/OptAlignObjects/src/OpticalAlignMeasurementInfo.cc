#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream & os, const OpticalAlignMeasurementInfo & meas)
{
  os << "Name: " << meas.name_ << " Type: " << meas.type_ << "  ID: " << meas.ID_ << std::endl;
  int iw = os.width(); // save current width
  int ip = os.precision(); // save current precision
  int now = 12;
  int nop = 5;

  std::vector<std::string>::iterator item;

  for(item = meas.measObjectNames_.begin(); item != meas.measObjectNames_.end(); item++ ){
    os << std::setw( now ) << std::setprecision( nop ) << "measuring object name: " << *item << std::endl;
  }

  std::vector<OpticalAlignParam>::iterator iteo;
  uint ii = 0;
  for( iteo = meas.values_.begin(); iteo != meas.values_.end(); iteo++ ){
    os << std::setw( now ) << std::setprecision( nop ) << "MEAS: " << " isSimu " << meas.isSimulatedValue_[ii++]  << *iteo;
  }
   return os;
}


