#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <iostream>

CSCReadoutMapping::CSCReadoutMapping() : myName_("CSCReadoutMapping"), debugV_( false ) {}

CSCReadoutMapping::~CSCReadoutMapping(){}

int CSCReadoutMapping::chamber( int endcap, int station, int vme, int dmb, int tmb ){
  // Build hw id from input, find sw id to match
  int cid = 0;
  int hid = hwId( endcap, station, vme, dmb, tmb );
  // Search for that hw id in mapping
  std::map<int,int>::iterator it = hw2sw_.find( hid );
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

CSCDetId CSCReadoutMapping::detId( int endcap, int station, int vme, int dmb, int tmb, int layer ){
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

int CSCReadoutMapping::hwId( int endcap, int station, int vmecrate, int dmb, int tmb ) {
  int id = 0;
  //@@ FIXME This is ONLY for Slice Test Nov-2005
  id = vmecrate * 16 + dmb;
  if ( debugV() ) std::cout << myName_ <<": hardware id for endcap " << endcap <<
    " station " << station << " vmecrate " << vmecrate << " dmb slot " << dmb <<
    " tmb slot " << tmb << " = " << id << std::endl;
  return id;
}

int CSCReadoutMapping::swId( int endcap, int station, int ring, int chamber ) {
  // Software id is just CSCDetId for the chamber
  return CSCDetId::rawIdMaker( endcap, station, ring, chamber, 0 ); // usual detid for chamber, i.e. layer=0
}
