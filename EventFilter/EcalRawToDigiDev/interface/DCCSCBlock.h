#ifndef DCCSCBLOCK_HH
#define DCCSCBLOCK_HH

#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "DCCFEBlock.h"

class DCCEventBlock;
class DCCDataUnpacker;

class DCCSCBlock : public DCCFEBlock {
	
  //to implement
	
  public :

    DCCSCBlock(DCCDataUnpacker * u, EcalElectronicsMapper *m, DCCEventBlock * e, bool unpack);
	 
	 void updateCollectors();
	 
	 
  protected :

   void unpackXtalData(uint stripID, uint xtalID);
	 
   EEDetId                                * pDetId_;
   EEDataFrame                            * pDFId_;
	 
   std::auto_ptr<EEDigiCollection>             * digis_;
	
	/* 
    todo : update this for the endcap...
	 
    auto_ptr<EEDetIdCollection>            * invalidGains_;  
    auto_ptr<EEDetIdCollection>            * invalidGainsSwitch_ ;
    auto_ptr<EEDetIdCollection>            * invalidChIds_;
	*/
    
};


#endif
