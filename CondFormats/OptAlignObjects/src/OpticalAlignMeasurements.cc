#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

std::ostream & operator<<(std::ostream & os, const OpticalAlignMeasurements & r)
{

  os << " There are " << r.oaMeasurements_.size() << " optical alignment objects." << std::endl;
  size_t max = r.oaMeasurements_.size();
  size_t xi = 0;
  while ( xi < max ) {
    os << "\t" << r.oaMeasurements_[xi];
    xi++;
  }
  return os;
}
