#include "CondFormats/OptAlignObjects/interface/XXXXMeasurements.h"

std::ostream & operator<<(std::ostream & os, const XXXXMeasurements & r)
{

  os << " There are " << r.xxxxMeasurements_.size() << " optical alignment objects." << std::endl;
  size_t max = r.opticalAlignments_.size();
  size_t oAi = 0;
  while ( oAi < max ) {
    os << "\t" << r.opticalAlignments_[oAi];
    oAi++;
  }
  return os;
}
