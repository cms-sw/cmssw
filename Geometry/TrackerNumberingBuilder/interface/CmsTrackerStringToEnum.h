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
  
  GeometricDet::GeometricEnumType type(std::string const&) const;
  std::string const & name(GeometricDet::GeometricEnumType) const;

 private:
  static MapEnumType const & map() { return m_impl._map;}
  static ReverseMapEnumType const & reverseMap() { return m_impl._reverseMap;}

  // a quick fix
  struct Impl {
    Impl();
    MapEnumType _map;
    ReverseMapEnumType _reverseMap;
  };

  static const Impl m_impl;

};
#endif
