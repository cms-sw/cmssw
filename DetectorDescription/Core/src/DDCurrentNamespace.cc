#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"

std::string & 
DDCurrentNamespace::ns() 
{
  static std::string ns_ = "GLOBAL";
  return ns_;
}
