#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"

CmsTrackerStringToEnum::Impl CmsTrackerStringToEnum::m_impl;

CmsTrackerStringToEnum::Impl::Impl(){
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("FullTracker",GeometricDet::Tracker));

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelBarrel",GeometricDet::PixelBarrel));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelBarrelLayer",GeometricDet::layer));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelBarrelLadder",GeometricDet::ladder));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelBarrelDet",GeometricDet::DetUnit));

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelEndcapSubDet",GeometricDet::PixelEndCap));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelEndcapDisk",GeometricDet::disk));  
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelEndcapPanel",GeometricDet::panel));  
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("PixelEndcapDet",GeometricDet::DetUnit)); 

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIB",GeometricDet::TIB));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIBLayer",GeometricDet::layer));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIBString",GeometricDet::strng));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIBGluedDet",GeometricDet::mergedDet));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIBDet",GeometricDet::DetUnit));

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TID",GeometricDet::TID));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIDWheel",GeometricDet::wheel));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIDRing",GeometricDet::ring));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIDGluedDet",GeometricDet::mergedDet));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TIDDet",GeometricDet::DetUnit));

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TOB",GeometricDet::TOB));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TOBLayer",GeometricDet::layer));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TOBRod",GeometricDet::rod));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TOBGluedDet",GeometricDet::mergedDet));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TOBDet",GeometricDet::DetUnit));

  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TEC",GeometricDet::TEC));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TECWheel",GeometricDet::wheel));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TECPetal",GeometricDet::petal));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TECRing",GeometricDet::ring));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TECGluedDet",GeometricDet::mergedDet));
  _map.insert(std::pair<std::string, GeometricDet::GeometricEnumType>("TECDet",GeometricDet::DetUnit));


  //
  // build reverse map
  //

  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::Tracker,"Tracker"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::PixelBarrel,"PixelBarrel"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::PixelEndCap,"PixelEndCap"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::TIB,"TIB"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::TID,"TID"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::TOB,"TOB"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::TEC,"TEC"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::layer,"layer"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::wheel,"Wheel"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::strng,"String"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::rod,"Rod"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::petal,"Petal"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::ring,"Ring"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::ladder,"Ladder"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::mergedDet,"GluedDet"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::DetUnit,"DetUnit"));
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::disk,"Disk")); 
  _reverseMap.insert(std::pair<GeometricDet::GeometricEnumType, std::string>(GeometricDet::panel,"Panel")); 

  //
  // done
  //
}

GeometricDet::GeometricEnumType CmsTrackerStringToEnum::type(std::string const & s) const{
  MapEnumType::const_iterator p=map().find(s);
  if (p!= map().end()) return p->second;
  return GeometricDet::unknown;
}

std::string const & CmsTrackerStringToEnum::name(GeometricDet::GeometricEnumType t) const {
  static std::string const u("Unknown");
  ReverseMapEnumType::const_iterator p=reverseMap().find(t); 
  if (p!= reverseMap().end())
    return p->second;
  return u;
}

