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

    DCCSCBlock(DCCDataUnpacker * u, EcalElectronicsMapper *m, DCCEventBlock * e, bool unpack, bool forceToKeepFRdata);
	 
    void updateCollectors();
	 
	 
  protected :

   int unpackXtalData(uint stripID, uint xtalID);
   void fillEcalElectronicsError( std::auto_ptr<EcalElectronicsIdCollection> * );
	 
   EEDetId                                * pDetId_;
   EEDataFrame                            * pDFId_;
	 
   std::auto_ptr<EEDigiCollection>        * digis_;

   // to restructure as common collections to DCCTowerBlock, to inherit from DCCFEBlock
   std::auto_ptr<EEDetIdCollection>       * invalidGains_;
   std::auto_ptr<EEDetIdCollection>       * invalidGainsSwitch_ ;
   std::auto_ptr<EEDetIdCollection>       * invalidChIds_;
    
};


#endif
