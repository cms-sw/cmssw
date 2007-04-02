
#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

/*
 *\ Class DCCTCCBlock
 *
 * Class responsible for the trigger primitives unpacking.
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2007/03/28 00:43:18 $
 * $Revision: 1.1.2.2 $
 *
 * \author N. Almeida
 * 
 *
*/

#include <iostream>                  
#include <string>
#include <vector>
#include <map>
#include <utility>


#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>

#include "DCCDataBlockPrototype.h"

class DCCDataUnpacker;

using namespace edm;

class DCCTCCBlock : public DCCDataBlockPrototype {
	
  public :
    /**
      Class constructor
    */
    DCCTCCBlock( DCCDataUnpacker *  u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack);    
   
    virtual void addTriggerPrimitivesToCollection(){};

    /**
      Unpacks TCC data 
     */
    void unpack(uint64_t ** data, uint * dwToEnd);
	 
    void display(ostream & o); 
	 
  
  protected :

    virtual void checkTccIdAndNumbTTs(){};
	  
    uint tccId_;
    uint bx_;
    uint l1_;
    uint nTTs_;
    uint nTSamples_;
    uint expNumbTTs_;
    uint expTccId_;
    
    EcalTrigTowerDetId * pTTDetId_;   
    EcalTriggerPrimitiveDigi * pTP_;
    auto_ptr<EcalTrigPrimDigiCollection > * tps_;  

};

#endif
