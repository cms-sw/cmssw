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
