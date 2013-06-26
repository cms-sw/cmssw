#include "CondFormats/CSCObjects/interface/CSCTriggerSimpleMapping.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include <iostream>
#include <fstream>
#include <sstream>

CSCTriggerSimpleMapping::CSCTriggerSimpleMapping(){}

CSCTriggerSimpleMapping::~CSCTriggerSimpleMapping(){}

int CSCTriggerSimpleMapping::hwId( int endcap, int station, int sector, int subsector, int cscid ) const {
 
  int id = 0;
  int ring = CSCTriggerNumbering::ringFromTriggerLabels(station,cscid);
  int chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,cscid);
  // This is ONLY for Slice Test Nov-2005
  
  id = CSCDetId::rawIdMaker(endcap,station,ring,chamber,0);
  
  if ( debugV() ) std::cout << myName() << ": hardware id for endcap " << endcap <<
    " station " << station << " sector " << sector << " subsector " << subsector <<
    " cscid " << cscid << " = " << id << std::endl;
  return id;
}
