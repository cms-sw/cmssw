#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

/*
 *\ Class DCCTCCBlock
 *
 * Class responsible for the trigger primitives unpacking.
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2008/11/18 12:36:00 $
 * $Revision: 1.8 $
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
#include <DataFormats/EcalDigi/interface/EcalPseudoStripInputDigi.h>
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
    int unpack(uint64_t ** data, uint * dwToEnd, short tccChId=0);
	 
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
    uint ps_;
    
    EcalTrigTowerDetId * pTTDetId_;   
    EcalTriggerPrimitiveDigi * pTP_;
    EcalPseudoStripInputDigi * pPS_;
    std::auto_ptr<EcalTrigPrimDigiCollection> * tps_;  
    std::auto_ptr<EcalPSInputDigiCollection> * pss_;  

};

#endif
