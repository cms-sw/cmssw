#ifndef DCCEESRPBLOCK_HH
#define DCCEESRPBLOCK_HH


/*
 *\ Class DCCEESRPBlock
 *
 * Class responsible for the SR flag unpacking in the EE.
 *
 * \file DCCEESRPBlock.h
 *
 * $Date: 2007/07/24 11:39:35 $
 * $Revision: 1.5 $
 *
 * \author N. Almeida
 *
*/


#include <iostream>
#include <memory>
#include <stdint.h>
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
    
    void updateCollectors();
	 
  protected :
  
    void addSRFlagToCollection(); 
    
    bool checkSrpIdAndNumbSRFlags();
	 
    std::auto_ptr<EESrFlagCollection>  * eeSrFlagsDigis_;
	 
    EcalScDetId * pSCDetId_;
    
	 
		
};


#endif
