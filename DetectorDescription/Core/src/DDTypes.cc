#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <iostream>
#include <utility>

using namespace geant_units;
using namespace geant_units::operators;


////////// output operator for printing the arguments of an algorithm

std::ostream & operator<<(std::ostream & os, const DDNumericArguments & t)
{
  DDNumericArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << '=' << it->second << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDStringArguments & t)
{
  DDStringArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << '=' << it->second << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDVectorArguments & t)
{
  DDVectorArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::vector<double>::const_iterator vit(it->second.begin()), ved(it->second.end());
    for(;vit!=ved;++vit) {
      os << *vit << ' ';
    }
    os << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDMapArguments & t)
{
  DDMapArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::map<std::string,double>::const_iterator mit(it->second.begin()), med(it->second.end());
    for(;mit!=med;++mit) {
      os << mit->first << '=' << mit->second << ' ';
    }
    os << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDStringVectorArguments & t)
{
  DDStringVectorArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::vector<std::string>::const_iterator vit(it->second.begin()), ved(it->second.end());
    for(; vit!=ved; ++vit) {
     os << *vit << ' '; 
    }
    os << std::endl;
  }
  return os;
}


// Formats an angle in radians as a 0-padded string in degrees; e.g. "-001.293900" for -1.2939 degrees.
std::string formatAsDegrees(double radianVal)
{
	const unsigned short numlen = 12;
	char degstr[numlen];
	int retval = snprintf(degstr, numlen, "%0*f", numlen - 1, convertRadToDeg( radianVal ));
	if (retval == numlen - 1)
		return degstr;
	else return "0000.000000";
}

// Formats an angle in radians as a 0-padded string in degrees expressed as integer between 0 and 360; e.g. "090" for -270.001 degrees.
std::string formatAsDegreesInInteger(double radianVal) {

  const unsigned short numlen = 4;
  char degstr[numlen];
  int degVal = std::lround(convertRadToDeg( radianVal ));
  if      (degVal <    0) degVal += 360;
  else if (degVal >= 360) degVal -= 360;
  int retval = snprintf(degstr, numlen, "%0*d", numlen-1, degVal);

  if (retval == numlen - 1) return degstr;
  else                      return "000";
}
