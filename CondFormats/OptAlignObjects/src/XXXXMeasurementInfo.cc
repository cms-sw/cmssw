#include "CondFormats/OptAlignObjects/interface/XXXXMeasurementInfo.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream & os, const XXXXMeasurementInfo & r)
{
  os << "Type: " << r.objectType_ << "  ID: " << r.objectID_ << std::endl;
  int iw = os.width(); // save current width
  int ip = os.precision(); // save current precision
  int now = 12;
  int nop = 5;
  os << std::setw( now ) << std::setprecision( nop ) << "member";
  os << std::setw( now ) << std::setprecision( nop ) << "value";
  os << std::setw( now ) << std::setprecision( nop ) << "error";
  os << std::setw( now ) << std::setprecision( nop ) << "qual." << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "x1" << r.x1_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "x2" << r.x2_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "x3" << r.x3_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "x4" << r.x4_ << std::endl;
  os <<  std::setw( now ) << std::setprecision( nop ) << "--- Extra Entries --- " << std::endl;
  size_t max = r.extraEntries_.size();
  size_t iE = 0;
  while ( iE < max ) {
    os << "[" << iE << "]" << r.extraEntries_[iE];
    iE++;
  }
  os << std::setprecision( ip ) << std::setw( iw );
  return os;
}


