#include "CondFormats/OptAlignObjects/interface/XXXXMeasurements.h"

std::ostream & operator<<(std::ostream & os, const XXXXMeasurements & r)
{

  os << " There are " << r.xxxxMeasurements_.size() << " optical alignment objects." << std::endl;
  size_t max = r.xxxxMeasurements_.size();
  size_t xi = 0;
  while ( xi < max ) {
    os << "\t" << r.xxxxMeasurements_[xi];
    xi++;
  }
  return os;
}
