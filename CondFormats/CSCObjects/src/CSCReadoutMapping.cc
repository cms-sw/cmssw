#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
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
    //    std::cout << "hwid = " << hid << ", swid = " << cid << std::endl;
    //    LogDebug("CSC") << " for requested hw id = " << hid << ", found sw id = " << cid;
  }
  else {
    edm::LogError("CSC") << " cannot find requested hw id = " << hid << " in mapping.";
  }
  return cid;
}

CSCDetId CSCReadoutMapping::detId( int endcap, int station, int vme, int dmb, int tmb, int cfeb, 
        int layer ) const {

  // Find CSCDetId index of chamber corresponding to the hardware readout arguments
  int cid = chamber( endcap, station, vme, dmb, tmb );

  // Decode the individual labels
  // ... include endcap & station for MTCC when they are unique in the mapping file
  // and so do not need to be specified as input arguments
      endcap  = CSCDetId::endcap( cid );
      station = CSCDetId::station( cid );
  int chamber = CSCDetId::chamber( cid );
  int ring    = CSCDetId::ring( cid );

  // Now sort out ME1a from ME11-combined
  // cfeb =0-3 for ME1b, cfeb=4 for ME1a
  if ( station == 1  && ring == 1 && cfeb == 4 ) {
      // This is ME1a region
      ring = 4; // reset from 1 to 4 which flags ME1a
  }
  return CSCDetId( endcap, station, ring, chamber, layer );
}

void CSCReadoutMapping::addRecord( int endcap, int station, int ring, int chamber, 
           int vmecrate, int dmb, int tmb, int tsector, int cscid ) {

  CSCLabel newRecord( endcap, station, ring, chamber, vmecrate, dmb, tmb, tsector, cscid );
  mapping_.push_back( newRecord );
  int hid = hwId( endcap, station, vmecrate, dmb, tmb );
  int sid = swId( endcap, station, ring, chamber);
  // LogDebug("CSC") << " map hw " << hid << " to sw " << sid;
  if ( hw2sw_.insert( std::make_pair(hid, sid) ).second ) {
    // LogDebug("CSC") << " insert pair succeeded.";
  }
  else {
    edm::LogError("CSC") << " already have key = " << hid;
  }
} 

int CSCReadoutMapping::swId( int endcap, int station, int ring, int chamber ) const {
  // Software id is just CSCDetId for the chamber - but no distinction within ME11
  return CSCDetId::rawIdMaker( endcap, station, ring, chamber, 0 ); // usual detid for chamber, i.e. layer=0
}

int CSCReadoutMapping::crate(const CSCDetId & id) const {
  return 1;//dummy
}
int CSCReadoutMapping::dmbId(const CSCDetId & id) const {
  return 1;//dummy
}
int CSCReadoutMapping::DDU(const CSCDetId & id) const {
  return 1;//dummy
}
int CSCReadoutMapping::DCC(const CSCDetId & id) const {
  return 1;//dummy
}
