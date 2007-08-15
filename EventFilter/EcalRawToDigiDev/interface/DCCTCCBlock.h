#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

/*
 *\ Class DCCTCCBlock
 *
 * Class responsible for the trigger primitives unpacking.
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2007/07/24 11:39:35 $
 * $Revision: 1.5 $
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
    int unpack(uint64_t ** data, uint * dwToEnd);
	 
    void display(std::ostream & o); 
	 
  
  protected :

    virtual bool checkTccIdAndNumbTTs(){return true;};
	  
    uint tccId_;
    uint bx_;
    uint l1_;
    uint nTTs_;
    uint nTSamples_;
    uint expNumbTTs_;
    uint expTccId_;
    
    EcalTrigTowerDetId * pTTDetId_;   
    EcalTriggerPrimitiveDigi * pTP_;
    std::auto_ptr<EcalTrigPrimDigiCollection > * tps_;  

};

#endif
