#ifndef DCCEESRPBLOCK_HH
#define DCCEESRPBLOCK_HH


/*
 *\ Class DCCEESRPBlock
 *
 * Class responsible for the SR flag unpacking in the EE.
 *
 * \file DCCEESRPBlock.h
 *
 *
 * \author N. Almeida
 *
*/


#include <iostream>
#include <memory>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "DCCSRPBlock.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>


class DCCEESRPBlock : public DCCSRPBlock{
	
  public :

    DCCEESRPBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack);
    
    void updateCollectors() override;
	 
  protected :
  
    void addSRFlagToCollection() override; 
    
    bool checkSrpIdAndNumbSRFlags() override;
	 
    std::unique_ptr<EESrFlagCollection>  * eeSrFlagsDigis_;
	 
    EcalScDetId * pSCDetId_;
    
	 
		
};


#endif
