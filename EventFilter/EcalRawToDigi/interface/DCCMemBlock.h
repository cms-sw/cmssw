#ifndef DCCMEMBLOCK_HH
#define DCCMEMBLOCK_HH

/*
 *\ Class DCCMemBlock
 *
 * Class responsible for MEMs unpacking 
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2012/08/06 21:51:35 $
 * $Revision: 1.3 $
 *
 * \author N. Almeida
 * \author G. Franzoni
 *
*/

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

class DCCMemBlock : public DCCDataBlockPrototype {
	
  public :

    DCCMemBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e);
	 
    virtual ~DCCMemBlock(){}
	 
    void updateCollectors();
    
    void display(std::ostream & o); 
    
    int unpack(const uint64_t ** data, unsigned int * dwToEnd, unsigned int expectedTowerID);   
    			
  protected :
	 
    void unpackMemTowerData();
    void fillPnDiodeDigisCollection();

    std::vector<short> pn_;

    unsigned int expTowerID_;
    unsigned int expXtalTSamples_;
    unsigned int kSamplesPerPn_;
	 
    unsigned int lastStripId_;
    unsigned int lastXtalId_;
    unsigned int lastTowerBeforeMem_;

    unsigned int towerId_;	
    unsigned int numbDWInXtalBlock_;
    unsigned int xtalBlockSize_;
    unsigned int nTSamples_; 
    unsigned int unfilteredTowerBlockLength_; 
   
    unsigned int bx_;
    unsigned int l1_;
	 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemChIds_;  
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemBlockSizes_; 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemTtIds_; 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemGains_;
    std::auto_ptr<EcalPnDiodeDigiCollection>     * pnDiodeDigis_;
	
};


#endif
