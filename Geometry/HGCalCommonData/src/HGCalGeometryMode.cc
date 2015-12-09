#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template<>
StringToEnumParser<HGCalGeometryMode::GeometryMode>::StringToEnumParser() {
  enumMap["HGCalGeometryMode::Square"]  = HGCalGeometryMode::Square;
  enumMap["HGCalGeometryMode::Hexagon"] = HGCalGeometryMode::Hexagon;
}

