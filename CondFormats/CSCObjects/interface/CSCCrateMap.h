#ifndef CSCCrateMap_h
#define CSCCrateMap_h

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <map>

class CSCCrateMap{
 public:
  CSCCrateMap();
  ~CSCCrateMap();  

  const CSCMapItem::MapItem& item( int key )const;

  typedef std::map< int,CSCMapItem::MapItem > CSCMap;
  CSCMap crate_map;
};

#endif
