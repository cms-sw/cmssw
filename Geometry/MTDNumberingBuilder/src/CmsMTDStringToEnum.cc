#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"

const CmsMTDStringToEnum::Impl CmsMTDStringToEnum::m_impl;

CmsMTDStringToEnum::Impl::Impl() {
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("FastTimerRegion", GeometricTimingDet::MTD));

  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("BarrelTimingLayer",
                                                                                  GeometricTimingDet::BTL));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Layer1", GeometricTimingDet::BTLLayer));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Rod1", GeometricTimingDet::BTLTray));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("BModule", GeometricTimingDet::BTLModule));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("SensorPackage",
                                                                                  GeometricTimingDet::BTLSensor));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Crystal", GeometricTimingDet::BTLCrystal));

  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("EndcapTimingLayer",
                                                                                  GeometricTimingDet::ETL));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Disc1", GeometricTimingDet::ETLDisc));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Disc2", GeometricTimingDet::ETLDisc));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Ring", GeometricTimingDet::ETLRing));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("EModule", GeometricTimingDet::ETLModule));
  _map.insert(
      std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Sensor", GeometricTimingDet::ETLSensor));
}

GeometricTimingDet::GeometricTimingEnumType CmsMTDStringToEnum::type(std::string const& s) const {
  // remove namespace if present
  std::string_view v = s;
  auto first = v.find_first_of(":");
  v.remove_prefix(std::min(first + 1, v.size()));
  MapEnumType::const_iterator p = map().find({v.data(), v.size()});
  if (p != map().end())
    return p->second;
  return GeometricTimingDet::unknown;
}
