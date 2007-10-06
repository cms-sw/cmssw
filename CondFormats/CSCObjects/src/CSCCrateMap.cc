#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"

CSCCrateMap::CSCCrateMap(){}

CSCCrateMap::~CSCCrateMap(){}

const CSCMapItem::MapItem& CSCCrateMap::item( int key ) const {
  return (crate_map.find(key))->second;
}
