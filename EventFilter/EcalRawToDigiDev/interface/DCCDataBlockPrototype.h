 #ifndef DCCDATABLOCKPROTOTYPE_HH
#define DCCDATABLOCKPROTOTYPE_HH

/*
 * \class DCCDataBlockPrototype
 * Prototype for ECAL data block unpacking
 * \file DCCDataBlockPrototype.h
 *
 * $Date: 2007/04/10 17:33:48 $
 * $Revision: 1.4 $
 * \author N. Almeida
 *
*/


#include <iostream>                  
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <stdio.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "DCCRawDataDefinitions.h"
#include <stdint.h>

class EcalElectronicsMapper;
class DCCDataUnpacker;
class DCCEventBlock;

class DCCDataBlockPrototype {
	
  public :
    /**
      Class constructor
    */
    DCCDataBlockPrototype( DCCDataUnpacker *  unpacker, EcalElectronicsMapper * mapper, DCCEventBlock * event, bool unpack = true);    
  
    virtual int unpack(uint64_t ** data, uint * dwToEnd){ return BLOCK_UNPACKED;}

    virtual void updateCollectors(){};
	
    virtual void display(std::ostream & o){} 

    void enableSyncChecks(){sync_=true;}
    
    /**
     Updates data pointer and dw to end of event
    */
    virtual void updateEventPointers(){ 

     //cout<<"\n block Length "<<blockLength_;
     //cout<<"\n dwToEne...   "<<*dwToEnd_;    

      *datap_   += blockLength_;
      *dwToEnd_ -= blockLength_; 
    }
    
    uint getLength(){ return blockLength_; }

  
  protected :
    DCCDataUnpacker       * unpacker_;
    bool error_; 
    EcalElectronicsMapper * mapper_;
    DCCEventBlock         * event_;
   
    
    uint64_t             ** datap_;
    uint64_t              * data_;
    uint                  * dwToEnd_;
   
   
    
    uint blockLength_;
    bool unpackInternalData_;
    bool sync_;

};

#endif
