#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"

const CmsMTDStringToEnum::Impl CmsMTDStringToEnum::m_impl;

CmsMTDStringToEnum::Impl::Impl() {
  map_.emplace("FastTimerRegion", GeometricTimingDet::MTD);
  map_.emplace("BarrelTimingLayer", GeometricTimingDet::BTL);
  map_.emplace("Layer1", GeometricTimingDet::BTLLayer);
  map_.emplace("Layer1Timing", GeometricTimingDet::BTLLayer);
  map_.emplace("BModule", GeometricTimingDet::BTLModule);
  map_.emplace("EndcapTimingLayer", GeometricTimingDet::ETL);
  map_.emplace("Disc1", GeometricTimingDet::ETLDisc);
  map_.emplace("Disc1Timing", GeometricTimingDet::ETLDisc);
  map_.emplace("Disc2Timing", GeometricTimingDet::ETLDisc);
  map_.emplace("EModule", GeometricTimingDet::ETLModule);
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
