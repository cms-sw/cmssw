#ifndef DCCMEMBLOCK_HH
#define DCCMEMBLOCK_HH

/*
 *\ Class DCCMemBlock
 *
 * Class responsible for MEMs unpacking 
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2007/10/23 14:17:53 $
 * $Revision: 1.8 $
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
    
    int unpack(uint64_t ** data, uint * dwToEnd, uint expectedTowerID);   
    			
  protected :
	 
    void unpackMemTowerData();
    void fillPnDiodeDigisCollection();

    std::vector<short> pn_;

    uint expTowerID_;
    uint expXtalTSamples_;
    uint kSamplesPerPn_;
	 
    uint lastStripId_;
    uint lastXtalId_;
    uint lastTowerBeforeMem_;

    uint towerId_;	
    uint numbDWInXtalBlock_;
    uint xtalBlockSize_;
    uint nTSamples_; 
    uint unfilteredTowerBlockLength_; 
   
    uint bx_;
    uint l1_;
	 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemChIds_;  
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemBlockSizes_; 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemTtIds_; 
    std::auto_ptr<EcalElectronicsIdCollection>   * invalidMemGains_;
    std::auto_ptr<EcalPnDiodeDigiCollection>     * pnDiodeDigis_;
	
};


#endif
