#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template<>
HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Hexagon"] = HGCalGeometryMode::Hexagon;
  enumMap["HGCalGeometryMode::HexagonFull"] = HGCalGeometryMode::HexagonFull;
}

template<>
HGCalStringToEnumParser<HGCalGeometryMode::WaferMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Polyhedra"]  = HGCalGeometryMode::Polyhedra;
  enumMap["HGCalGeometryMode::ExtrudedPolygon"] = HGCalGeometryMode::ExtrudedPolygon;
}

