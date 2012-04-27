#ifndef DDAxes_h
#define DDAxes_h

#include "DetectorDescription/Base/interface/Singleton.h"

#include <string>
#include <map>

//! analagous to geant4/source/global/HEPGeometry/include/geomdefs.hh
enum DDAxes {x, y, z, rho, radial3D, phi, undefined};

class AxesNames
{
  
public:
  AxesNames();
  ~AxesNames();
  
  const std::string name(const DDAxes& s) ;
  
  DDAxes index(const std::string & s);
  
private:
  std::map<std::string, DDAxes> axesmap_;
};
  

class DDAxesNames : public DDI::Singleton<AxesNames>
{
public:

  static const std::string name(const DDAxes& s);

  static DDAxes index(const std::string & s);
};	   

#endif // DDAxes_h
