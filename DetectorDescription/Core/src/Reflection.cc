#include "DetectorDescription/Core/src/Reflection.h"

#include <utility>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

DDI::Reflection::Reflection(const DDSolid & s)
 : Solid(ddreflected), s_(s)
{ }

double DDI::Reflection::volume() const
{
  return s_.isDefined().second ?  s_.volume() : -1.;    
}

void DDI::Reflection::stream(std::ostream & os) const
{
  os << " reflection solid of " << s_;
}
