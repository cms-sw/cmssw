#include "DetectorDescription/Core/src/Solid.h"
#include <ostream>

void DDI::Solid::stream(std::ostream & os) const
{
  for( const auto& i : p_ ) 
    os << i << ' ';
}
