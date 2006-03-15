#include "Geometry/TrackerGeometryBuilder/interface/GeomDetTypeIdToEnum.h"


GeomDetTypeIdToEnum::GeomDetTypeIdToEnum(){
  _map.clear();
  _reverseMap.clear();
  //
  // Insert !
  //

  _map.insert(std::pair<int, GeomDetType::SubDetector>(1,GeomDetType::PixelBarrel));
  _map.insert(std::pair<int, GeomDetType::SubDetector>(2,GeomDetType::PixelEndcap));
  _map.insert(std::pair<int, GeomDetType::SubDetector>(3,GeomDetType::TIB));
  _map.insert(std::pair<int, GeomDetType::SubDetector>(4,GeomDetType::TID));
  _map.insert(std::pair<int, GeomDetType::SubDetector>(5,GeomDetType::TOB));
  _map.insert(std::pair<int, GeomDetType::SubDetector>(6,GeomDetType::TEC));


  //
  // build reverse map
  //

  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::PixelBarrel,1));
  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::PixelEndcap,2));
  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::TIB,3));
  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::TID,4));
  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::TOB,5));
  _reverseMap.insert(std::pair<GeomDetType::SubDetector, int>(GeomDetType::TEC,6));

  //
  // done
  //
}
GeomDetType::SubDetector& GeomDetTypeIdToEnum::type(int s){
  if (_map.find(s) != _map.end())
    return (_map.find(s))->second;
  //return GeomDetType::unknown;
}

int GeomDetTypeIdToEnum::detId(GeomDetType::SubDetector t){
  if (_reverseMap.find(t) != _reverseMap.end())
    return (_reverseMap.find(t))->second;
  return 100;
}


