#include <CondFormats/CSCObjects/interface/CSCTriggerMapping.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <iostream>

CSCTriggerMapping::CSCTriggerMapping() : myName_("CSCTriggerMapping"), debugV_( false ) {}

CSCTriggerMapping::~CSCTriggerMapping(){}

int CSCTriggerMapping::chamber( int endcap, int station, int sector, int subsector, int cscid ) const {
  // Build hw id from input, find sw id to match
  int cid = 0;
  int hid = hwId( endcap, station, sector, subsector, cscid );
  // Search for that hw id in mapping
  std::map<int,int>::const_iterator it = hw2sw_.find( hid );
  if ( it != hw2sw_.end() ) {
    cid = it->second;
    if ( debugV() ) std::cout << myName_ << ": for requested hw id = " << hid <<
       ", found sw id = " << cid << std::endl;
  }
  else {
    std::cout << myName_ << ": ERROR, cannot find requested hw id = " << hid <<
      " in mapping." << std::endl;
  }
  return cid;
}

CSCDetId CSCTriggerMapping::detId( int endcap, int station, int sector, int subsector, int cscid, int layer ) const {
  int cid = chamber( endcap, station, sector, subsector, cscid );
  int lid = cid + layer;
  return CSCDetId( lid );
}

void CSCTriggerMapping::addRecord( int rendcap, int rstation, int rsector, int rsubsector, int rcscid, 
				   int cendcap, int cstation, int csector, int csubsector, int ccscid ) {

  Connection newRecord( rendcap, rstation, rsector, rsubsector, rcscid, cendcap, cstation, csector, csubsector, ccscid );
  mapping_.push_back( newRecord );
  int hid = hwId( rendcap, rstation, rsector, rsubsector, rcscid );
  int sid = swId( cendcap, cstation, csector, csubsector, ccscid );
  if ( debugV() ) std::cout << myName_ << ": map hw " << hid << " to sw " << sid << std::endl;
  if ( hw2sw_.insert( std::make_pair( hid, sid) ).second ) {
    if ( debugV() ) std::cout << myName_ << ": insert pair succeeded." << std::endl;
  }
  else {
    std::cout << myName_ << ": ERROR, already have key = " << hid << std::endl;
  }
} 

int CSCTriggerMapping::swId( int endcap, int station, int sector, int subsector, int cscid ) const {
  // Software id is just CSCDetId for the chamber
  int ring = CSCTriggerNumbering::ringFromTriggerLabels(station, cscid);
  int chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,cscid);
  return CSCDetId::rawIdMaker( endcap, station, ring, chamber, 0 ); // usual detid for chamber, i.e. layer=0
}
