#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"

CSCChamberMap::CSCChamberMap(){}

CSCChamberMap::~CSCChamberMap(){}

const CSCMapItem::MapItem& CSCChamberMap::item( int key ) const { 
  return (ch_map.find(key))->second; 
}
