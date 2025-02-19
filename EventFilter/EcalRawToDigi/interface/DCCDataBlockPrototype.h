#ifndef DCCDATABLOCKPROTOTYPE_HH
#define DCCDATABLOCKPROTOTYPE_HH

/*
 * \class DCCDataBlockPrototype
 * Prototype for ECAL data block unpacking
 * \file DCCDataBlockPrototype.h
 *
 * $Date: 2012/08/06 21:51:35 $
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

    virtual ~DCCDataBlockPrototype() {};
  
    virtual int unpack(const uint64_t ** data, unsigned int * dwToEnd){ return BLOCK_UNPACKED;}

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

      // preventing pointers from navigating wildly outside of fedBlock
      if((*dwToEnd_)>=blockLength_) 
        *dwToEnd_ -= blockLength_; 
      else 
        *dwToEnd_ = 0; 

    }
    
    virtual unsigned int getLength(){ return blockLength_; }

  
  protected :
    DCCDataUnpacker       * unpacker_;
    bool error_; 
    EcalElectronicsMapper * mapper_;
    DCCEventBlock         * event_;
   
    
    const uint64_t             ** datap_;
    const uint64_t              * data_;
    unsigned int                  * dwToEnd_;
   
   
    unsigned int blockLength_;
    bool unpackInternalData_;
    bool sync_;

};

#endif
