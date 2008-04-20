#ifndef DCCEVENTBLOCK_HH
#define DCCEVENTBLOCK_HH


/*
 *\ Class DCCEventBlock
 *
 * Class responsible for managing the raw data unpacking.
 * The class instantes the DCCMemBlock 
 *
 * \file DCCEventBlock.h
 *
 * $Date: 2007/08/15 14:23:28 $
 * $Revision: 1.7 $
 *
 * \author N. Almeida
 * \author G. Franzoni
 *
*/

#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include "DCCRawDataDefinitions.h"

class DCCFEBlock;
class DCCTCCBlock;
class DCCSRPBlock;
class DCCDataUnpacker;
class DCCMemBlock;
class EcalElectronicsMapper;


class DCCEventBlock {
	
  public :

   DCCEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, bool hU, bool srpU, bool tccU, bool feU, bool memU);
	
   virtual ~DCCEventBlock();  
 
   void unpack( uint64_t * buffer, uint bufferSize, uint expFedId);
	
   void enableSyncChecks();

   void updateCollectors();
	
   void display(std::ostream & o);
		
   uint smId()                  { return smId_;     }
   uint l1A()                   { return l1_;       }
   uint bx()                    { return bx_;       }
   DCCDataUnpacker  * unpacker(){ return unpacker_; }

    	
  protected :
     
    void addHeaderToCollection();
	 
    int virtual unpackTCCBlocks(){ return BLOCK_UNPACKED;}
 
    DCCDataUnpacker  *  unpacker_;
    uint64_t         *  data_; 
    uint eventSize_;
    uint dwToEnd_;
   
    std::vector<short> feChStatus_;
    std::vector<short> tccChStatus_;
    
    uint srChStatus_;

    uint fedId_;
    uint bx_;
    uint l1_;
    uint triggerType_;
    uint smId_;
    uint blockLength_;  
    uint dccErrors_;
    uint runNumber_;
    uint runType_;
    uint detailedTriggerType_;
    
    uint orbitCounter_;
    uint sr_;
    uint zs_;
    uint tzs_;
    
    DCCFEBlock             * towerBlock_;
    DCCTCCBlock            * tccBlock_;
    DCCMemBlock            * memBlock_;
    DCCSRPBlock            * srpBlock_;
    EcalElectronicsMapper  * mapper_;
	 
    bool headerUnpacking_;
    bool srpUnpacking_;
    bool tccUnpacking_;
    bool feUnpacking_;
    bool memUnpacking_;

    std::auto_ptr<EcalRawDataCollection> *  dccHeaders_;
	 

};

#endif
