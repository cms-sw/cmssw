#include <CondFormats/CSCObjects/interface/CSCReadoutAtlas.h>

#include <CondFormats/CSCObjects/interface/CSCCrateMap.h>
#include <CondFormats/DataRecord/interface/CSCCrateMapRcd.h>

#include <CondFormats/CSCObjects/interface/CSCChamberMap.h>
#include <CondFormats/DataRecord/interface/CSCChamberMapRcd.h>

#include <DataFormats/MuonDetId/interface/CSCIndexer.h>

#include <FWCore/Framework/interface/ESHandle.h>

CSCReadoutAtlas::CSCReadoutAtlas() : myName_("CSCReadoutAtlas"), debugV_( false ) {}

CSCReadoutAtlas::~CSCReadoutAtlas(){}

const CSCCrateMap* CSCReadoutAtlas::updateCrateMap( const edm::EventSetup& setup ) const {
  edm::ESHandle<CSCCrateMap> h;
  setup.get<CSCCrateMapRcd>().get( h );
  return h.product(); 
}

const CSCChamberMap* CSCReadoutAtlas::updateChamberMap( const edm::EventSetup& setup ) const {
  edm::ESHandle<CSCChamberMap> h;
  setup.get<CSCChamberMapRcd>().get( h );
  return h.product(); 
}

int CSCReadoutAtlas::dbIndex( const CSCDetId& id ) const {
  CSCIndexer indexer;
  int dummy = 0;
  return indexer.dbIndex(id, dummy ); // 2nd arg dummy in this context
}

CSCDetId CSCReadoutAtlas::detId( const edm::EventSetup& setup, int vme, int dmb, int cfeb, int il ) const {

  const CSCCrateMap* pcrate = updateCrateMap( setup );

  int cscid = dmb;
  if ( dmb >= 6 ) --cscid;
  int key = vme*10 + cscid;
  CSCMapItem::MapItem mitem = pcrate->item( key );

  int ie = mitem.endcap;
  int is = mitem.station;
  int ir = mitem.ring;
  int ic = mitem.chamber;

  // Now sort out ME1a from ME11-combined
  // cfeb =0-3 for ME1b, cfeb=4 for ME1a
  if ( is == 1  && ir == 1 && cfeb == 4 ) {
      // This is ME1a region
      ir = 4; // reset from 1 to 4 which flags ME1a
  }
  return CSCDetId( ie, is, ir, ic, il );}

int CSCReadoutAtlas::crate( const edm::EventSetup& setup, const CSCDetId & id ) const {
  int igor = dbIndex( id );
  const CSCChamberMap* p = updateChamberMap( setup );
  CSCMapItem::MapItem mitem = p->item( igor );  
  return mitem.crateid;
}
int CSCReadoutAtlas::dmb( const edm::EventSetup& setup, const CSCDetId & id ) const {
  int igor = dbIndex( id );
  const CSCChamberMap* p = updateChamberMap( setup );
  CSCMapItem::MapItem mitem = p->item( igor );  
  return mitem.dmb;
}
int CSCReadoutAtlas::ddu( const edm::EventSetup& setup, const CSCDetId & id ) const {
  int igor = dbIndex( id );
  const CSCChamberMap* p = updateChamberMap( setup );
  CSCMapItem::MapItem mitem = p->item( igor );  
  return mitem.ddu;
}
int CSCReadoutAtlas::slink( const edm::EventSetup& setup, const CSCDetId & id ) const {
  int igor = dbIndex( id );
  const CSCChamberMap* p = updateChamberMap( setup );
  CSCMapItem::MapItem mitem = p->item( igor );  
  return mitem.slink;
}
