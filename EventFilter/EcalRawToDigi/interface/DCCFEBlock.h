#ifndef DCCFEBLOCK_HH
#define DCCFEBLOCK_HH

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

#include "DCCDataBlockPrototype.h"

class DCCEventBlock;
class DCCDataUnpacker;

class DCCFEBlock : public DCCDataBlockPrototype {
	
  public :

    DCCFEBlock(DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack, bool forceToKeepFRdata);
    
    virtual ~DCCFEBlock(){ delete [] xtalGains_;}

    void zsFlag(bool zs){ zs_ = zs;}

    void enableFeIdChecks(){checkFeId_= true;}
	 
    virtual void updateCollectors();
    
    void display(std::ostream & o); 
    using DCCDataBlockPrototype::unpack; 
    int unpack(const uint64_t** data, unsigned int * dwToEnd, bool zs, unsigned int expectedTowerID);

    unsigned int getLength(){return blockLength_; }
    			
  protected :
	 
    virtual int unpackXtalData(unsigned int stripID, unsigned int xtalID){      return BLOCK_UNPACKED;};
    virtual void fillEcalElectronicsError( std::unique_ptr<EcalElectronicsIdCollection> * ){};
    
    
    bool zs_;
    bool checkFeId_;
    unsigned int expTowerID_;
    bool forceToKeepFRdata_;
    unsigned int expXtalTSamples_;
    unsigned int unfilteredDataBlockLength_;
    unsigned int lastStripId_;
    unsigned int lastXtalId_;
 
    unsigned int towerId_;	
    unsigned int numbDWInXtalBlock_;
    unsigned int xtalBlockSize_;
    unsigned int nTSamples_; 
    
    unsigned int blockSize_;
    unsigned int bx_;
    unsigned int l1_;
    
    short * xtalGains_;
    std::unique_ptr<EcalElectronicsIdCollection> * invalidTTIds_;
    std::unique_ptr<EcalElectronicsIdCollection> * invalidZSXtalIds_;
    std::unique_ptr<EcalElectronicsIdCollection> * invalidBlockLengths_;

};


#endif
