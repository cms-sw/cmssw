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
  static constexpr size_t kModStrLen = 7;

  typedef std::map<std::string, GeometricTimingDet::GeometricTimingEnumType> MapEnumType;

  GeometricTimingDet::GeometricTimingEnumType type(std::string const&) const;

private:
  static MapEnumType const& map() { return m_impl._map; }

  // a quick fix
  struct Impl {
    Impl();
    MapEnumType _map;
  };

  static const Impl m_impl;
};
#endif
