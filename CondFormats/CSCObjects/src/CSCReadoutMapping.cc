#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <iostream>

CSCReadoutMapping::CSCReadoutMapping() : myName_("CSCReadoutMapping"), debugV_( false ) {}

CSCReadoutMapping::~CSCReadoutMapping(){}

int CSCReadoutMapping::chamber( int endcap, int station, int vme, int dmb, int tmb ) const {
  // Build hw id from input, find sw id to match
  int cid = 0;
  int hid = hwId( endcap, station, vme, dmb, tmb );
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

CSCDetId CSCReadoutMapping::detId( int endcap, int station, int vme, int dmb, int tmb, int layer ) const {
  int cid = chamber( endcap, station, vme, dmb, tmb );
  int lid = cid + layer;
  return CSCDetId( lid );
}

void CSCReadoutMapping::addRecord( int endcap, int station, int ring, int chamber, 
           int vmecrate, int dmb, int tmb, int tsector, int cscid ) {

  CSCLabel newRecord( endcap, station, ring, chamber, vmecrate, dmb, tmb, tsector, cscid );
  mapping_.push_back( newRecord );
  int hid = hwId( endcap, station, vmecrate, dmb, tmb );
  int sid = swId( endcap, station, ring, chamber);
  if ( debugV() ) std::cout << myName_ << ": map hw " << hid << " to sw " << sid << std::endl;
  if ( hw2sw_.insert( std::make_pair( hid, sid) ).second ) {
    if ( debugV() ) std::cout << myName_ << ": insert pair succeeded." << std::endl;
  }
  else {
    std::cout << myName_ << ": ERROR, already have key = " << hid << std::endl;
  }
} 

int CSCReadoutMapping::swId( int endcap, int station, int ring, int chamber ) const {
  // Software id is just CSCDetId for the chamber
  return CSCDetId::rawIdMaker( endcap, station, ring, chamber, 0 ); // usual detid for chamber, i.e. layer=0
}
