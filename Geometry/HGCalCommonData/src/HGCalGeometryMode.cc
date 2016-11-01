#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template<>
HGCalStringToEnumParser<HGCalGeometryMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Square"]  = HGCalGeometryMode::Square;
  enumMap["HGCalGeometryMode::Hexagon"] = HGCalGeometryMode::Hexagon;
  enumMap["HGCalGeometryMode::HexagonFull"] = HGCalGeometryMode::HexagonFull;
}

