#include <CondFormats/CSCObjects/interface/CSCChamberMap.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

CSCChamberMap::CSCChamberMap() {}

CSCChamberMap::~CSCChamberMap() {}

const CSCMapItem::MapItem& CSCChamberMap::item(int key) const { return (ch_map.find(key))->second; }

int CSCChamberMap::dbIndex(const CSCDetId& id) const {
  int ie = id.endcap();
  int is = id.station();
  int ir = id.ring();
  int ic = id.chamber();
  //  int il = id.layer(); // zero for parent chamber

  // ME1a must be reset to ME11
  if ((is == 1) && (ir == 4))
    ir = 1;

  return ie * 100000 + is * 10000 + ir * 1000 + ic * 10;
}

int CSCChamberMap::crate(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.crateid;
}

int CSCChamberMap::dmb(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.dmb;
}

int CSCChamberMap::ddu(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.ddu;
}

int CSCChamberMap::slink(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.slink;
}

int CSCChamberMap::dduSlot(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.ddu_slot;
}

int CSCChamberMap::dduInput(const CSCDetId& id) const {
  int igor = dbIndex(id);
  CSCMapItem::MapItem mitem = this->item(igor);
  return mitem.ddu_input;
}
