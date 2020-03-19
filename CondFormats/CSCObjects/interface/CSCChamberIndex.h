#ifndef CSCChamberIndex_h
#define CSCChamberIndex_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <vector>

class CSCChamberIndex {
public:
  CSCChamberIndex();
  ~CSCChamberIndex();

  const CSCMapItem::MapItem& item(int key) const;

  typedef std::vector<CSCMapItem::MapItem> CSCVector;
  CSCVector ch_index;

  COND_SERIALIZABLE;
};

#endif
