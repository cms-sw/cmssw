#include "DetectorDescription/Core/src/Solid.h"
#include<ostream>

void DDI::Solid::stream(std::ostream & os) const
{
  std::vector<double>::const_iterator i = p_.begin();
  std::vector<double>::const_iterator e = p_.end();
  for (; i != e ; ++i) 
    os << *i << ' ';
}
