#ifndef DCCEETCCBLOCK_HH
#define DCCEETCCBLOCK_HH

/*
 *\ Class DCCEETCCBlock
 *
 * Class responsible for the EE Trigger Tower primitives unpacking.
 *
 * \file DCCEETCCBlock.h
 *
 * $Date: 2008/12/11 18:05:56 $
 * $Revision: 1.1 $
 *
 * \author N. Almeida
 *
*/

#include <iostream>                  
#include <string>
#include <vector>
#include <map>
#include <utility>


#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include "DCCTCCBlock.h"

class DCCEETCCBlock : public DCCTCCBlock{
	
  public :
    /**
      Class constructor
    */
    DCCEETCCBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpacking );    
  
    void updateCollectors();
	 
    void addTriggerPrimitivesToCollection();
	
	uint getLength();
  
  protected :
  
    bool checkTccIdAndNumbTTs();


};

#endif
