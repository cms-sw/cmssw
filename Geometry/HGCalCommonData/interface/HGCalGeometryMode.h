#ifndef Geometry_HGCalCommonData_HGCalGeometryMode_H
#define Geometry_HGCalCommonData_HGCalGeometryMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

template< typename T >
class HGCalStringToEnumParser {
  std::map<std::string, T> enumMap;
public:
    
  HGCalStringToEnumParser(void);

  T parseString(const std::string &value)  { 
    typename std::map<std::string, T>::const_iterator itr = enumMap.find(value);
    if (itr == enumMap.end())
      throw cms::Exception("Configuration") << "the value " << value 
					    << " is not defined.";
    return itr->second;
  }
};

namespace HGCalGeometryMode {
  enum GeometryMode { Square=0, Hexagon=1, HexagonFull=2 };
  enum WaferMode { Polyhedra=0, ExtrudedPolygon=1};
}

#endif
