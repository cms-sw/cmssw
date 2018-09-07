#ifndef Geometry_MTDNumberingBuilder_CmsMTDStringToEnum_H
#define Geometry_MTDNumberingBuilder_CmsMTDStringToEnum_H

#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include <string>
#include <map>
/**
 * Builds map between Det type and an enum
 */
class CmsMTDStringToEnum {
 public:
  typedef std::map<std::string, GeometricTimingDet::GeometricTimingEnumType> MapEnumType;
  typedef std::map<GeometricTimingDet::GeometricTimingEnumType, std::string> ReverseMapEnumType;
  
  GeometricTimingDet::GeometricTimingEnumType type(std::string const&) const;
  std::string const & name(GeometricTimingDet::GeometricTimingEnumType) const;

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
