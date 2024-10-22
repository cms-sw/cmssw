#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"

CSCChamberIndex::CSCChamberIndex() {}

CSCChamberIndex::~CSCChamberIndex() {}

const CSCMapItem::MapItem& CSCChamberIndex::item(int key) const { return ch_index[key]; }
