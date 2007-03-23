#ifndef DCCMEMBLOCK_HH
#define DCCMEMBLOCK_HH

/*
 *\ Class DCCMemBlock
 *
 * Class responsible for MEMs unpacking 
 *
 * \file DCCTCCBlock.h
 *
 * $Date: 2007/03/20 00:50:12 $
 * $Revision: 1.1.2.1 $
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

using namespace std;
using namespace edm;

class DCCEventBlock;
class DCCDataUnpacker;

class DCCMemBlock : public DCCDataBlockPrototype {
	
  public :

    DCCMemBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e);
	 
    virtual ~DCCMemBlock(){}
	 
    void updateCollectors();
    
    void display(ostream & o); 
    
    void unpack(uint64_t ** data, uint * dwToEnd, uint expectedTowerID);     
     
    			
  protected :
	 
    void unpackMemTowerData();
    void fillPnDiodeDigisCollection();
    vector<short> pn_;

    uint expTowerID_;
    uint expXtalTSamples_;
    uint kSamplesPerPn_;
	 
    uint lastStripId_;
    uint lastXtalId_;
 
    uint towerId_;	
    uint numbDWInXtalBlock_;
    uint xtalBlockSize_;
    uint nTSamples_; 
    uint unfilteredTowerBlockLength_; 
   
    uint bx_;
    uint l1_;
	 
    auto_ptr<EcalElectronicsIdCollection>   * invalidMemChIds_;  
    auto_ptr<EcalElectronicsIdCollection>   * invalidMemBlockSizes_; 
    auto_ptr<EcalElectronicsIdCollection>   * invalidMemTtIds_; 
    auto_ptr<EcalElectronicsIdCollection>   * invalidMemGains_;
    auto_ptr<EcalPnDiodeDigiCollection>     * pnDiodeDigis_;
	
};


#endif
