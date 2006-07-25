#include "Geometry/TrackerGeometryBuilder/interface/GeomDetTypeIdToEnum.h"

using namespace GeomDetEnumerators;


GeomDetTypeIdToEnum::GeomDetTypeIdToEnum(){
  _map.clear();
  _reverseMap.clear();
  //
  // Insert !
  //

  _map.insert(std::pair<int, SubDetector>(1,PixelBarrel));
  _map.insert(std::pair<int, SubDetector>(2,PixelEndcap));
  _map.insert(std::pair<int, SubDetector>(3,TIB));
  _map.insert(std::pair<int, SubDetector>(4,TID));
  _map.insert(std::pair<int, SubDetector>(5,TOB));
  _map.insert(std::pair<int, SubDetector>(6,TEC));


  //
  // build reverse map
  //

  _reverseMap.insert(std::pair<SubDetector, int>(PixelBarrel,1));
  _reverseMap.insert(std::pair<SubDetector, int>(PixelEndcap,2));
  _reverseMap.insert(std::pair<SubDetector, int>(TIB,3));
  _reverseMap.insert(std::pair<SubDetector, int>(TID,4));
  _reverseMap.insert(std::pair<SubDetector, int>(TOB,5));
  _reverseMap.insert(std::pair<SubDetector, int>(TEC,6));

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


