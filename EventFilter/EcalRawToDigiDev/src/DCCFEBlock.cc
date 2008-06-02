
#include "EventFilter/EcalRawToDigiDev/interface/DCCFEBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include <stdio.h>
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"



DCCFEBlock::DCCFEBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, DCCEventBlock * e,bool unpack)
: DCCDataBlockPrototype(u,m,e,unpack){
   
  expXtalTSamples_           = mapper_->numbXtalTSamples();
  numbDWInXtalBlock_         = (expXtalTSamples_-2)/4+1;
  unfilteredDataBlockLength_ = mapper_->getUnfilteredTowerBlockLength();
  xtalGains_                 = new short[expXtalTSamples_]; 
  
}


void DCCFEBlock::updateCollectors(){

  invalidBlockLengths_    = unpacker_->invalidBlockLengthsCollection();
  invalidTTIds_           = unpacker_->invalidTTIdsCollection();

}



int DCCFEBlock::unpack(uint64_t ** data, uint * dwToEnd, bool zs, uint expectedTowerID){
  
  zs_      = zs;  
  datap_   = data;
  data_    = *data;
  dwToEnd_ = dwToEnd;
  
 
  if( (*dwToEnd_)<1){
   edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlockk::unpack")
      <<"\n Unable to unpack Tower block for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
      <<"\n The end of event was reached !";
    //TODO : add this to a dcc event size collection error?
    return STOP_EVENT_UNPACKING;
  }
  
  lastStripId_     = 0;
  lastXtalId_      = 0;
  expTowerID_      = expectedTowerID;
  
  
  //Point to begin of block
  data_++;
  
  towerId_           = ( *data_ )                 & TOWER_ID_MASK;
  nTSamples_         = ( *data_>>TOWER_NSAMP_B  ) & TOWER_NSAMP_MASK; 
  bx_                = ( *data_>>TOWER_BX_B     ) & TOWER_BX_MASK;
  l1_                = ( *data_>>TOWER_L1_B     ) & TOWER_L1_MASK;
  blockLength_       = ( *data_>>TOWER_LENGTH_B ) & TOWER_LENGTH_MASK;
  
  //debugging
  //display(cout);

  //check expected trigger tower id

  if( expTowerID_ != towerId_){

   edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock::unpack")
      <<"\n For event "<<event_->l1A()<<" and dcc "<<mapper_->getActiveDCC()
      <<"\n Expected trigger tower is "<<expTowerID_<<" while "<<towerId_<<" was found "
      <<"\n => Skipping this event...";
    
    //EcalTrigTowerDetId * tp = mapper_->getTTDetIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
    // this needs understanding and fix
    // (*invalidTTIds_)->push_back(*tp);

    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;     
       

  }
  
  // Check synchronization
  if(sync_){
    uint dccBx = (event_->bx())&TCC_BX_MASK;
    uint dccL1 = (event_->l1A())&TCC_L1_MASK; 
    if( dccBx != bx_ || dccL1 != l1_ ){
      edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock::unpack")
       <<"\n Synchronization error for Tower Block "<<towerId_<<" in event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in dcc "<<mapper_->getActiveDCC()
       <<"\n TCC local l1A is  "<<l1_<<" and local bx is "<<bx_
       <<"\n => Skipping this event...";
      //Note : add to error collection ?		 
      updateEventPointers();
	  return SKIP_BLOCK_UNPACKING;
	  
    }
  }

  // check number of samples
  if( nTSamples_ != expXtalTSamples_ ){
    edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock::unpack")
      <<"EcalRawToDigi@SUB=DCCFEBlock::unpack"
      <<"\n Unable to unpack Tower Block "<<towerId_<<" for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
      <<"\n Number of time samples "<<nTSamples_<<" is not the same as expected ("<<expXtalTSamples_<<")"
      <<"\n => Skipping this event...";
    //Note : add to error collection ?		 
    updateEventPointers();
	return SKIP_BLOCK_UNPACKING;
	
  }  
  
  xtalBlockSize_     = numbDWInXtalBlock_*8;
  blockSize_         = blockLength_*8;  
  
  if((*dwToEnd_)<blockLength_){
    
	edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock::unpack")
      <<"\n Unable to unpack Tower Block "<<towerId_<<" for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
      <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<blockSize_<<" are needed!"
      <<"\n => Skipping this event...";
    //TODO : add to error collections
    return STOP_EVENT_UNPACKING;
  }
  
  
  if(!zs_){
	 
    if ( unfilteredDataBlockLength_ != blockLength_ ){
      edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock")
        <<"\n For event "<<event_->l1A()<<", dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
        <<"\n Expected block size is "<<(unfilteredDataBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found"
        <<"\n => Skipping this event...";
     
      EcalTrigTowerDetId *  tp = mapper_->getTTDetIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
      (*invalidBlockLengths_)->push_back(*tp);

      //Safer approach... 
	  return STOP_EVENT_UNPACKING;	  
	
     }
  }else if( blockLength_ > unfilteredDataBlockLength_ || (blockLength_-1) < numbDWInXtalBlock_ ){
   
     edm::LogWarning("EcalRawToDigi@SUB=DCCFEBlock::unpack")
       <<"\n For event "<<event_->l1A()<<" and dcc "<<mapper_->getActiveDCC()
       <<"\n The tower "<<towerId_<<" has a wrong number of bytes : "<<(blockLength_*8)	   
       <<"\n => Skipping this event...";
      
      EcalTrigTowerDetId *  tp = mapper_->getTTDetIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
      (*invalidBlockLengths_)->push_back(*tp);
  
     //Safer approach... 
	 return STOP_EVENT_UNPACKING;
    
  }
  
  uint numbOfXtalBlocks = (blockLength_-1)/numbDWInXtalBlock_; 

  // get XTAL Data
  uint stripID(0), xtalID(0);
 
  //point to xtal data
  data_++;
  
  for(uint numbXtal=1; numbXtal <= numbOfXtalBlocks; numbXtal++){

    // If zs is disabled we know the expected strip and xtal ids
    // Note : this is valid for the EB how about the EE ? -> retieve expected index from mapper
	 
    if(!zs_){
      stripID =( numbXtal-1)/5 + 1;	
      xtalID  = numbXtal - (stripID-1)*5;
    }

    unpackXtalData(stripID,xtalID);
   
  }
  
  updateEventPointers();
  return BLOCK_UNPACKED;		
        
}




void DCCFEBlock::display(std::ostream& o){

  o<<"\n Unpacked Info for DCC Tower Block"
  <<"\n DW1 ============================="
  <<"\n Tower Id "<<towerId_
  <<"\n Numb Samp "<<nTSamples_
  <<"\n Bx "<<bx_
  <<"\n L1 "<<l1_
  <<"\n blockLength "<<blockLength_;  
} 


