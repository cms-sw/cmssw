#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

template <>
HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Square"] = HGCalGeometryMode::Square;
  enumMap["HGCalGeometryMode::Hexagon"] = HGCalGeometryMode::Hexagon;
  enumMap["HGCalGeometryMode::HexagonFull"] = HGCalGeometryMode::HexagonFull;
  enumMap["HGCalGeometryMode::Hexagon8"] = HGCalGeometryMode::Hexagon8;
  enumMap["HGCalGeometryMode::Hexagon8Full"] = HGCalGeometryMode::Hexagon8Full;
  enumMap["HGCalGeometryMode::Trapezoid"] = HGCalGeometryMode::Trapezoid;
  enumMap["HGCalGeometryMode::Hexagon8File"] = HGCalGeometryMode::Hexagon8File;
  enumMap["HGCalGeometryMode::TrapezoidFile"] = HGCalGeometryMode::TrapezoidFile;
  enumMap["HGCalGeometryMode::Hexagon8Module"] = HGCalGeometryMode::Hexagon8Module;
  enumMap["HGCalGeometryMode::TrapezoidModule"] = HGCalGeometryMode::TrapezoidModule;
  enumMap["HGCalGeometryMode::Hexagon8Cassette"] = HGCalGeometryMode::Hexagon8Cassette;
  enumMap["HGCalGeometryMode::TrapezoidCassette"] = HGCalGeometryMode::TrapezoidCassette;
}

template <>
HGCalStringToEnumParser<HGCalGeometryMode::WaferMode>::HGCalStringToEnumParser() {
  enumMap["HGCalGeometryMode::Polyhedra"] = HGCalGeometryMode::Polyhedra;
  enumMap["HGCalGeometryMode::ExtrudedPolygon"] = HGCalGeometryMode::ExtrudedPolygon;
}
