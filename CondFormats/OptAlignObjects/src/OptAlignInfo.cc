#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream & os, const OpticalAlignInfo & r)
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
  os << std::setw( now ) << std::setprecision( nop ) << "x" << r.x_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "y" << r.y_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "z" << r.z_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "angx" << r.angx_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "angy" << r.angy_ << std::endl;
  os << std::setw( now ) << std::setprecision( nop ) << "angz" << r.angz_ << std::endl;
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


std::ostream & operator<<(std::ostream & os, const OpticalAlignParam & r)
{
  int iw = std::cout.width(); // save current width
  int ip = std::cout.precision(); // save current precision
  int now = 12;
  int nop = 5;
  os << std::setw( now ) << std::setprecision( nop ) << r.value_;
  os << std::setw( now ) << std::setprecision( nop ) << r.error_;
  os << std::setw( now ) << std::setprecision( nop ) << r.qual_ << std::endl;

  // Reset the values we changed
  std::cout << std::setprecision( ip ) << std::setw( iw );
  return os;
}
