#include <CondFormats/GEMObjects/interface/GEMChamberMap.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

GEMChamberMap::GEMChamberMap(){}

GEMChamberMap::~GEMChamberMap(){}

const GEMMapItem::MapItem& GEMChamberMap::item( int key ) const { 
  return (ch_map.find(key))->second; 
}

int GEMChamberMap::dbIndex(const GEMDetId& id ) const {

  int ie = 2; // endcap + is 2 and - is 1
  if (ie < id.region()) ie = 1;
  int is = id.station();
  int ir = 1;//id.ring(); only in ring 1
  int ic = id.chamber();
  int il = id.layer();

  return ie*100000 + is*10000 + ir*1000 + ic*10 + il;
}

int GEMChamberMap::crate( const GEMDetId& id ) const {
  int igor = dbIndex( id );  
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.crateid;   
}

int GEMChamberMap::dmb( const GEMDetId& id ) const {
  int igor = dbIndex( id );  
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.dmb;   
}

int GEMChamberMap::ddu( const GEMDetId& id ) const {
  int igor = dbIndex( id );  
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.ddu;   
}

int GEMChamberMap::slink( const GEMDetId& id ) const {
  int igor = dbIndex( id );  
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.slink;   
}

int GEMChamberMap::dduSlot( const GEMDetId& id ) const {
  int igor = dbIndex( id );
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.ddu_slot;
}

int GEMChamberMap::dduInput( const GEMDetId& id ) const {
  int igor = dbIndex( id );
  GEMMapItem::MapItem mitem = this->item( igor );
  return mitem.ddu_input;
}
