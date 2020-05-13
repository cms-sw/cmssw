#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"

const CmsMTDStringToEnum::Impl CmsMTDStringToEnum::m_impl;

CmsMTDStringToEnum::Impl::Impl() {
  _map.emplace("FastTimerRegion", GeometricTimingDet::MTD);
  _map.emplace("BarrelTimingLayer", GeometricTimingDet::BTL);
  _map.emplace("Layer1", GeometricTimingDet::BTLLayer);
  _map.emplace("Layer1Timing", GeometricTimingDet::BTLLayer);
  _map.emplace("BModule", GeometricTimingDet::BTLModule);
  _map.emplace("EndcapTimingLayer", GeometricTimingDet::ETL);
  _map.emplace("Disc1", GeometricTimingDet::ETLDisc);
  _map.emplace("Disc1Timing", GeometricTimingDet::ETLDisc);
  _map.emplace("Disc2Timing", GeometricTimingDet::ETLDisc);
  _map.emplace("EModule", GeometricTimingDet::ETLModule);
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
