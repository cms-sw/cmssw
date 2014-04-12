#include <CondFormats/CSCObjects/interface/CSCCrateMap.h>

CSCCrateMap::CSCCrateMap(){}

CSCCrateMap::~CSCCrateMap(){}

const CSCMapItem::MapItem& CSCCrateMap::item( int key ) const {
  return (crate_map.find(key))->second;
}

CSCDetId CSCCrateMap::detId( int vme, int dmb, int cfeb, int layer ) const {
  int cscid = dmb; 
  if ( dmb >= 6 ) --cscid; 
  int key = vme*10 + cscid; 
  const CSCMapItem::MapItem& mitem = this->item( key ); 
  int ie = mitem.endcap; 
  int is = mitem.station; 
  int ir = mitem.ring;
  int ic = mitem.chamber;

  // Now sort out ME1a from ME11-combined
  // cfeb =0-3 for ME1b, cfeb=4 for ME1a (pre-LS1) cfeb=4-6 for ME1a (post-LS1)
  if ( is == 1  && ir == 1 && cfeb >= 4 && cfeb <=6 ) { 
    // This is ME1a region  
    ir = 4; // reset from 1 to 4 which flags ME1a 
  }
  return CSCDetId( ie, is, ir, ic, layer );
} 
