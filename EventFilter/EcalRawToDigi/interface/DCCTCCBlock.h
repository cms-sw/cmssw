#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

/*
 *\ Class DCCTCCBlock
 *
 * Class responsible for the trigger primitives unpacking.
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2012/08/06 21:51:35 $
 * $Revision: 1.3 $
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
    int unpack(const uint64_t ** data, unsigned int * dwToEnd, short tccChId=0);
	 
    void display(std::ostream & o); 
	 
  
  protected :

    virtual bool checkTccIdAndNumbTTs(){return true;};
	  
    unsigned int tccId_;
    unsigned int bx_;
    unsigned int l1_;
    unsigned int nTTs_;
    unsigned int nTSamples_;
    unsigned int expNumbTTs_;
    unsigned int expTccId_;
    unsigned int ps_;
    
    EcalTrigTowerDetId * pTTDetId_;   
    EcalTriggerPrimitiveDigi * pTP_;
    EcalPseudoStripInputDigi * pPS_;
    std::auto_ptr<EcalTrigPrimDigiCollection> * tps_;  
    std::auto_ptr<EcalPSInputDigiCollection> * pss_;  

};

#endif
