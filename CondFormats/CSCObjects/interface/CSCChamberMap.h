#ifndef CSCChamberMap_h
#define CSCChamberMap_h

#include "OnlineDB/CSCCondDB/interface/CSCMapItem.h"
#include <map>

class CSCChamberMap{
 public:
  CSCChamberMap();
  ~CSCChamberMap();

  typedef std::map< int,CSCMapItem::MapItem > CSCMap;
  CSCMap ch_map;
};

#endif
