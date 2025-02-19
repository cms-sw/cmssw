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
 * $Date: 2012/08/22 01:11:31 $
 * $Revision: 1.8 $
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

   DCCEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, bool hU, bool srpU, bool tccU, bool feU, bool memU, bool forceToKeepFRdata);
	
   virtual ~DCCEventBlock();  
 
   virtual void unpack(const uint64_t * buffer, size_t bufferSize, unsigned int expFedId){};
   
   void reset();
	
   void enableSyncChecks();
	
   void enableFeIdChecks();

   void updateCollectors();
	
   void display(std::ostream & o);
		
   unsigned int smId()                  { return smId_;     }
   unsigned int fov()                   { return fov_;      }
   unsigned int mem()                   { return mem_;      }
   unsigned int l1A()                   { return l1_;       }
   unsigned int bx()                    { return bx_;       }
   DCCDataUnpacker  * unpacker(){ return unpacker_; }
   
   void setSRPSyncNumbers(short l1, short bx){ srpLv1_=l1; srpBx_=bx; }
   void setFESyncNumbers(short l1, short bx, short id){ feLv1_[id]= l1; feBx_[id]=bx;}
   void setTCCSyncNumbers(short l1, short bx, short id){ tccLv1_[id]= l1; tccBx_[id]=bx;}
   void setHLTChannel( int channel, short value ){ hlt_[channel-1] = value; }   
   short getHLTChannel(int channel){ return hlt_[channel-1];}

    	
  protected :
     
    void addHeaderToCollection();
	 
    int virtual unpackTCCBlocks(){ return BLOCK_UNPACKED;}
 
    DCCDataUnpacker  *  unpacker_;
    const uint64_t       *  data_; 
    unsigned int eventSize_;
    unsigned int dwToEnd_;
    
    unsigned int next_tower_search(const unsigned int current_tower_id);
    
    std::vector<short> feChStatus_;
    std::vector<short> tccChStatus_;
    std::vector<short> hlt_;

    std::vector<short> feLv1_; std::vector<short> feBx_;  
    std::vector<short> tccLv1_; std::vector<short> tccBx_;    
    short srpLv1_; short srpBx_; 

    
    unsigned int srChStatus_;

    unsigned int fov_;
    unsigned int fedId_;
    unsigned int bx_;
    unsigned int l1_;
    unsigned int triggerType_;
    unsigned int smId_;
    unsigned int blockLength_;  
    unsigned int dccErrors_;
    unsigned int runNumber_;
    unsigned int runType_;
    unsigned int detailedTriggerType_;
    
    unsigned int orbitCounter_;
    unsigned int mem_;
    unsigned int sr_;
    unsigned int zs_;
    unsigned int tzs_;
    
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
    bool forceToKeepFRdata_;

    std::auto_ptr<EcalRawDataCollection> *  dccHeaders_;
	 

};


// this code intended for sync checking in files:
//   DCC(FE|Mem|TCC|SRP)Block.cc

enum BlockType {FE_MEM = 1, TCC_SRP = 2};

bool isSynced(const unsigned int dccBx,
              const unsigned int bx,
              const unsigned int dccL1,
              const unsigned int l1,
              const BlockType type,
              const unsigned int fov);

#endif
