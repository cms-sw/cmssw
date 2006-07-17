#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream & os, const OpticalAlignInfo & r)
{
  os << "Name: " << r.objectName_ << std::endl;
  os << "Parent Name: " << r.parentObjectName_ << std::endl; 
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
  os << std::setw( now ) << std::setprecision( nop ) << r.name_;
  os << std::setw( now ) << std::setprecision( nop ) << r.dimension_;
  os << std::setw( now ) << std::setprecision( nop ) << r.value_;
  os << std::setw( now ) << std::setprecision( nop ) << r.error_;
  os << std::setw( now ) << std::setprecision( nop ) << r.qual_ << std::endl;

  // Reset the values we changed
  std::cout << std::setprecision( ip ) << std::setw( iw );
  return os;
}

//   // copy constructor and assignment operator
// OpticalAlignParam::OpticalAlignParam ( OpticalAlignParam& rhs ) {
//   name_ = rhs.name_;
//   value_ = rhs.value_;
//   error_ = rhs.error_;
//   qual_ = rhs.qual_;
//   dimension_ = rhs.dimension_;
// }

// OpticalAlignParam::OpticalAlignParam ( const OpticalAlignParam& rhs ) {
//   name_ = rhs.name_;
//   value_ = rhs.value_;
//   error_ = rhs.error_;
//   qual_ = rhs.qual_;
//   dimension_ = rhs.dimension_;
// }

// OpticalAlignInfo::OpticalAlignInfo ( OpticalAlignInfo& rhs ) {
//   x_ = rhs.x_;
//   y_ = rhs.y_;
//   z_ = rhs.z_;
//   angx_ = rhs.angx_;
//   angy_ = rhs.angy_;
//   angz_ = rhs.angz_;
//   std::vector<OpticalAlignParam>::const_iterator oapit = rhs.extraEntries_.begin();
//   std::vector<OpticalAlignParam>::const_iterator oapendit = rhs.extraEntries_.end();
//   if ( oapit == oapendit ) {
//     extraEntries_.clear();
//   } else {
//     for ( ; oapit != oapendit; ++oapit ) {
//       extraEntries_.push_back (*oapit);
//     }
//   }
//   objectType_ = rhs.objectType_;
//   objectName_ = rhs.objectName_;
//   parentObjectName_ = rhs.parentObjectName_;
// }

// OpticalAlignInfo::OpticalAlignInfo ( const OpticalAlignInfo& rhs ) {
//   x_ = rhs.x_;
//   y_ = rhs.y_;
//   z_ = rhs.z_;
//   angx_ = rhs.angx_;
//   angy_ = rhs.angy_;
//   angz_ = rhs.angz_;
//   std::vector<OpticalAlignParam>::const_iterator oapit = rhs.extraEntries_.begin();
//   std::vector<OpticalAlignParam>::const_iterator oapendit = rhs.extraEntries_.end();
//   if ( oapit == oapendit ) {
//     extraEntries_.clear();
//   } else {
//     for ( ; oapit != oapendit; ++oapit ) {
//       extraEntries_.push_back (*oapit);
//     }
//   }
//   objectType_ = rhs.objectType_;
//   objectName_ = rhs.objectName_;
//   parentObjectName_ = rhs.parentObjectName_;
// }
