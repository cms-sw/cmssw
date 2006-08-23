#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"

std::ostream & operator<<(std::ostream & os, const OpticalAlignments & r)
{

  os << " There are " << r.opticalAlignments_.size() << " optical alignment objects." << std::endl;
  size_t max = r.opticalAlignments_.size();
  size_t oAi = 0;
  while ( oAi < max ) {
    os << "\t" << r.opticalAlignments_[oAi];
    oAi++;
  }
  return os;
}

// OpticalAlignments::OpticalAlignments ( OpticalAlignments& rhs ) {
//   std::vector<OpticalAlignInfo>::const_iterator oait = rhs.opticalAlignments_.begin();
//   std::vector<OpticalAlignInfo>::const_iterator oaendit = rhs.opticalAlignments_.end();
//   if ( oait == oaendit ) {
//     opticalAlignments_.clear();
//   } else {
//     for ( ; oait != oaendit; ++oait) {
//       opticalAlignments_.push_back(*oait);
//     }
//   }
// }

// OpticalAlignments::OpticalAlignments ( const OpticalAlignments& rhs ) {
//   std::vector<OpticalAlignInfo>::const_iterator oait = rhs.opticalAlignments_.begin();
//   std::vector<OpticalAlignInfo>::const_iterator oaendit = rhs.opticalAlignments_.end();
//   if ( oait == oaendit ) {
//     opticalAlignments_.clear();
//   } else {
//     for ( ; oait != oaendit; ++oait) {
//       opticalAlignments_.push_back(*oait);
//     }
//   }
// }
