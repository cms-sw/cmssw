#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DetectorDescription/Core/interface/DDutils.h"

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

HGCalGeometryMode::GeometryMode HGCalGeometryMode::getGeometryMode(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<std::string>& fvec = val.strings();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "getGeometryMode::Failed to get " << s << " tag.";
    }

    HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
    HGCalGeometryMode::GeometryMode result = (HGCalGeometryMode::GeometryMode)eparser.parseString(fvec[0]);
    return result;
  } else {
    throw cms::Exception("HGCalGeom") << "getGeometryMode::Failed to fetch " << s << " tag";
  }
};

HGCalGeometryMode::GeometryMode HGCalGeometryMode::getGeometryMode(const std::string& s) {
  HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
  HGCalGeometryMode::GeometryMode result = (HGCalGeometryMode::GeometryMode)eparser.parseString(s);
  return result;
};

HGCalGeometryMode::WaferMode HGCalGeometryMode::getGeometryWaferMode(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<std::string>& fvec = val.strings();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "getGeometryWaferMode::Failed to get " << s << " tag.";
    }

    HGCalStringToEnumParser<HGCalGeometryMode::WaferMode> eparser;
    HGCalGeometryMode::WaferMode result = (HGCalGeometryMode::WaferMode)eparser.parseString(fvec[0]);
    return result;
  } else {
    throw cms::Exception("HGCalGeom") << "getGeometryWaferMode::Failed to fetch " << s << " tag";
  }
};

HGCalGeometryMode::WaferMode HGCalGeometryMode::getGeometryWaferMode(std::string& s) {
  HGCalStringToEnumParser<HGCalGeometryMode::WaferMode> eparser;
  HGCalGeometryMode::WaferMode result = (HGCalGeometryMode::WaferMode)eparser.parseString(s);
  return result;
};
