#ifndef DCCTowerBLOCK_HH
#define DCCTowerBLOCK_HH

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

using namespace std;
using namespace edm;

class DCCEventBlock;
class DCCDataUnpacker;

class DCCTowerBlock : public DCCFEBlock {
	
  //to implement
	
  public :

    DCCTowerBlock(DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack );
    
	 void updateCollectors();
	 
  protected:
	 
    void unpackXtalData(uint stripID, uint xtalID);

    auto_ptr<EBDigiCollection>     * digis_;
    
    EBDetId                        * pDetId_;
    EBDataFrame                    * pDFId_;

    auto_ptr<EBDetIdCollection>    * invalidGains_;  
    auto_ptr<EBDetIdCollection>    * invalidGainsSwitch_ ;
    auto_ptr<EBDetIdCollection>    * invalidGainsSwitchStay_;
    auto_ptr<EBDetIdCollection>    * invalidChIds_;
	 
};


#endif
