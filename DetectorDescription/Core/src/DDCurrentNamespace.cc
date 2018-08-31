#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"

std::ostream & operator<<( std::ostream & os, const DDCurrentNamespace & ns ) {
  return os << *ns;
}
