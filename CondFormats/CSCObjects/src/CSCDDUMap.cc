#include "CondFormats/CSCObjects/interface/CSCDDUMap.h"

CSCDDUMap::CSCDDUMap(){}

CSCDDUMap::~CSCDDUMap(){}

const CSCMapItem::MapItem& CSCDDUMap::item( int key ) const { 
  return (ddu_map.find(key))->second; 
}
