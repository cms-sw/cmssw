#ifndef DDAxes_h
#define DDAxes_h

#include <string>

//! analagous to geant4/source/global/HEPGeometry/include/geomdefs.hh
enum DDAxes {x, y, z, rho, radial3D, phi, undefined};

class DDAxesNames
{
public:
  static char const*name(const DDAxes& s);
  static DDAxes index(const std::string &s);
};	   

#endif // DDAxes_h
