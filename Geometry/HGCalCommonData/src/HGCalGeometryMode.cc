#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template<>
HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Square"]       = HGCalGeometryMode::Square;
  enumMap["HGCalGeometryMode::Hexagon"]      = HGCalGeometryMode::Hexagon;
  enumMap["HGCalGeometryMode::HexagonFull"]  = HGCalGeometryMode::HexagonFull;
  enumMap["HGCalGeometryMode::Hexagon8"]     = HGCalGeometryMode::Hexagon8;
  enumMap["HGCalGeometryMode::Hexagon8Full"] = HGCalGeometryMode::Hexagon8Full;
  enumMap["HGCalGeometryMode::Trapezoid"]    = HGCalGeometryMode::Trapezoid;
 }

template<>
HGCalStringToEnumParser<HGCalGeometryMode::WaferMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Polyhedra"]       = HGCalGeometryMode::Polyhedra;
  enumMap["HGCalGeometryMode::ExtrudedPolygon"] = HGCalGeometryMode::ExtrudedPolygon;
}

