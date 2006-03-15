#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerStringToEnum_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerStringToEnum_H

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include <string>
#include <map>
/**
 * Builds map between Det type and an enum
 */
class CmsTrackerStringToEnum {
 public:
  typedef std::map<std::string, GeometricDet::GeometricEnumType> MapEnumType;
  typedef std::map<GeometricDet::GeometricEnumType, std::string> ReverseMapEnumType;

  CmsTrackerStringToEnum();
  
  GeometricDet::GeometricEnumType type(std::string);
  std::string name(GeometricDet::GeometricEnumType);

 private:
  MapEnumType _map;
  ReverseMapEnumType _reverseMap;

};
#endif
