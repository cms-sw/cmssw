#ifndef DCCSRPBLOCK_HH
#define DCCSRPBLOCK_HH


/*
 *\ Class DCCSRPBlock
 *
 * Class responsible for SR flag unpacking.
 *
 * \file DCCSRPBlock.h
 *
 * $Date: 2008/12/11 18:05:57 $
 * $Revision: 1.1 $
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

#include "DCCDataBlockPrototype.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>


class DCCSRPBlock : public DCCDataBlockPrototype {
	
  public :

    DCCSRPBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack);
	 
    void display(std::ostream & o); 

    int unpack(uint64_t ** data, unsigned int * dwToEnd, unsigned int numbFlags = SRP_NUMBFLAGS);     	 

    ushort srFlag(unsigned int feChannel){ return srFlags_[feChannel-1]; }
    			
  protected :
    
    virtual void addSRFlagToCollection(){};
	 
    virtual bool checkSrpIdAndNumbSRFlags(){ return true; };
	 
    unsigned int srpId_         ;  
    unsigned int bx_            ;  
    unsigned int l1_            ;   
    unsigned int nSRFlags_      ; 
    unsigned int expNumbSrFlags_;
	 
    ushort srFlags_[SRP_NUMBFLAGS]; 
	 
	 
	 
};


#endif
