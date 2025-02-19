#ifndef DCCSRPBLOCK_HH
#define DCCSRPBLOCK_HH


/*
 *\ Class DCCSRPBlock
 *
 * Class responsible for SR flag unpacking.
 *
 * \file DCCSRPBlock.h
 *
 * $Date: 2012/08/06 21:51:35 $
 * $Revision: 1.4 $
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

    int unpack(const uint64_t ** data, unsigned int * dwToEnd, unsigned int numbFlags = SRP_NUMBFLAGS);     	 

    unsigned short srFlag(unsigned int feChannel){ return srFlags_[feChannel-1]; }
    			
  protected :
    
    virtual void addSRFlagToCollection(){};
	 
    virtual bool checkSrpIdAndNumbSRFlags(){ return true; };
	 
    unsigned int srpId_         ;  
    unsigned int bx_            ;  
    unsigned int l1_            ;   
    unsigned int nSRFlags_      ; 
    unsigned int expNumbSrFlags_;
	 
    unsigned short srFlags_[SRP_NUMBFLAGS]; 
	 
	 
	 
};


#endif
