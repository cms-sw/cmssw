#ifndef DCCTowerBLOCK_HH
#define DCCTowerBLOCK_HH

#include <iostream>
#include <memory>
#include <cstdint>
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

class DCCTowerBlock : public DCCFEBlock {
	
  //to implement
	
  public :

    DCCTowerBlock(DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack, bool forceToKeepFRdata );
    
    void updateCollectors() override;
	 
  protected:
	 
    int unpackXtalData(unsigned int stripID, unsigned int xtalID) override;
    void fillEcalElectronicsError( std::unique_ptr<EcalElectronicsIdCollection> * ) override;

    std::unique_ptr<EBDigiCollection>     * digis_;
    
    EBDetId                             * pDetId_;

    // to restructure as common collections to DCCSCBlock, to inherit from DCCFEBlock
    std::unique_ptr<EBDetIdCollection>    * invalidGains_;  
    std::unique_ptr<EBDetIdCollection>    * invalidGainsSwitch_ ;
    std::unique_ptr<EBDetIdCollection>    * invalidChIds_;
	 
};


#endif
