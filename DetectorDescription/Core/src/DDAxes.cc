#include "DetectorDescription/Core/interface/DDAxes.h"

AxesNames::AxesNames()
  : axesmap_{{"x", DDAxes::x }, {"y", DDAxes::y}, {"z", DDAxes::z}, {"rho", DDAxes::rho}, {"radial3D", DDAxes::radial3D}, {"phi", DDAxes::phi}, {"undefined", DDAxes::undefined }}
{}

AxesNames::~AxesNames() { }

const std::string
AxesNames::name(const DDAxes& s) 
{
  for( const auto& it : axesmap_ )
  {
    if( it.second == s )
      return it.first;
  }
  return "undefined";
}

const std::string
DDAxesNames::name(const DDAxes& s) 
{
  return instance().name(s);
}
