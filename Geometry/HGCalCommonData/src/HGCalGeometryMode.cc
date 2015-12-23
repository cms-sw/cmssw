#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template<>
HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Square"]  = HGCalGeometryMode::Square;
  enumMap["HGCalGeometryMode::Hexagon"] = HGCalGeometryMode::Hexagon;
}

