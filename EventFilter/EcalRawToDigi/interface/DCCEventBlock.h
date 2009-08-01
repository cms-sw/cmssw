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
 * $Date: 2009/03/12 10:06:15 $
 * $Revision: 1.2 $
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
 
   virtual void unpack( uint64_t * buffer, uint bufferSize, uint expFedId){};
   
   void reset();
	
   void enableSyncChecks();
	
   void enableFeIdChecks();

   void updateCollectors();
	
   void display(std::ostream & o);
		
   uint smId()                  { return smId_;     }
   uint fov()                   { return fov_;      }
   uint mem()                   { return mem_;      }
   uint l1A()                   { return l1_;       }
   uint bx()                    { return bx_;       }
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
    uint64_t         *  data_; 
    uint eventSize_;
    uint dwToEnd_;
   
    std::vector<short> feChStatus_;
    std::vector<short> tccChStatus_;
    std::vector<short> hlt_;

    std::vector<short> feLv1_; std::vector<short> feBx_;  
    std::vector<short> tccLv1_; std::vector<short> tccBx_;    
    short srpLv1_; short srpBx_; 

    
    uint srChStatus_;

    uint fov_;
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
    uint mem_;
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
    bool forceToKeepFRdata_;

    std::auto_ptr<EcalRawDataCollection> *  dccHeaders_;
	 

};

#endif
