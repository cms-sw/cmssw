#ifndef CSCDDUMap_h
#define CSCDDUMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <map>

class CSCDDUMap {
public:
  CSCDDUMap();
  ~CSCDDUMap();

  const CSCMapItem::MapItem& item(int key) const;

  typedef std::map<int, CSCMapItem::MapItem> CSCMap;
  CSCMap ddu_map;

  COND_SERIALIZABLE;
};

#endif
