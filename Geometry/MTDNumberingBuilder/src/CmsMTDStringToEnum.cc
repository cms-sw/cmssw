#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"

const CmsMTDStringToEnum::Impl CmsMTDStringToEnum::m_impl;

CmsMTDStringToEnum::Impl::Impl(){
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("FastTimerRegion",GeometricTimingDet::MTD));

  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("btl:BarrelTimingLayer",GeometricTimingDet::BTL));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:Layer1",GeometricTimingDet::BTLLayer));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:Rod1",GeometricTimingDet::BTLTray));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:BModule",GeometricTimingDet::BTLModule));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("SensorPackage",GeometricTimingDet::BTLSensor));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Crystal",GeometricTimingDet::BTLCrystal));

  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("etl:EndcapTimingLayer",GeometricTimingDet::ETL));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:Disc1",GeometricTimingDet::ETLDisc));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:Ring",GeometricTimingDet::ETLRing));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("mtd:EModule",GeometricTimingDet::ETLModule));
  _map.insert(std::pair<std::string, GeometricTimingDet::GeometricTimingEnumType>("Sensor",GeometricTimingDet::ETLSensor));
  
  //
  // build reverse map
  //

  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::MTD,"FastTimerRegion"));
  
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTL,"btl:BarrelTimingLayer"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTLLayer,"mtd:Layer"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTLTray,"mtd:Rod1"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTLModule,"Module"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTLSensor,"SensorPackage"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::BTLCrystal,"Crystal"));
  
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::ETL,"etl:EndcapTimingLayer"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::ETLDisc,"mtd:Disc1"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::ETLRing,"mtd:Ring"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::ETLModule,"Module"));
  _reverseMap.insert(std::pair<GeometricTimingDet::GeometricTimingEnumType, std::string>(GeometricTimingDet::ETLSensor,"Sensor"));
  
  //
  // done
  //
}

GeometricTimingDet::GeometricTimingEnumType CmsMTDStringToEnum::type(std::string const & s) const{
  MapEnumType::const_iterator p=map().find(s);
  if (p!= map().end()) return p->second;
  return GeometricTimingDet::unknown;
}

std::string const & CmsMTDStringToEnum::name(GeometricTimingDet::GeometricTimingEnumType t) const {
  static std::string const u("Unknown");
  ReverseMapEnumType::const_iterator p=reverseMap().find(t); 
  if (p!= reverseMap().end())
    return p->second;
  return u;
}

