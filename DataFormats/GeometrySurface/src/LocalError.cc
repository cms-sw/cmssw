#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include <iostream>

std::ostream & operator<<( std::ostream& s, const LocalError& err) {
  return s << " (" << err.xx() << "," << err.xy() << "," << err.yy() << ") ";
}


