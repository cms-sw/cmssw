#ifndef Geometry_TrackerGeometryBuilder_GeomDetTypeIdToEnum_H
#define Geometry_TrackerGeometryBuilder_GeomDetTypeIdToEnum_H

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include <map>

/**
 * Builds map between DetId and an enum
 */
class GeomDetTypeIdToEnum {
 public:
  typedef std::map<int, GeomDetType::SubDetector> MapEnumType;
  typedef std::map<GeomDetType::SubDetector, int> ReverseMapEnumType;

  GeomDetTypeIdToEnum();
  
  GeomDetType::SubDetector& type(int);
  int detId(GeomDetType::SubDetector);

 private:
  MapEnumType _map;
  ReverseMapEnumType _reverseMap;

};
#endif
