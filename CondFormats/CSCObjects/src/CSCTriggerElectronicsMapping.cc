#include "CondFormats/CSCObjects/interface/CSCTriggerElectronicsMapping.h"
#include <iostream>
#include <fstream>
#include <sstream>

CSCTriggerElectronicsMapping::CSCTriggerElectronicsMapping(){}

CSCTriggerElectronicsMapping::~CSCTriggerElectronicsMapping(){}

int CSCTriggerElectronicsMapping::hwId( int SPboardID, int FPGA, int cscid, int zero1, int zero2 ) const {
 
  int id = 0;
    
  id = ((cscid) | (FPGA << 4) | (SPboardID << 8));
  
  if ( debugV() ) std::cout << myName() << ": hardware id for SP Board Id " << SPboardID 
			    << " FPGA " << FPGA << " cscid " << cscid  << " = " << id << std::endl;
  return id;
}
