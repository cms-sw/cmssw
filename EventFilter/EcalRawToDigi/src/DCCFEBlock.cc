#include "EventFilter/EcalRawToDigi/interface/DCCFEBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include <stdio.h>
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"



DCCFEBlock::DCCFEBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, DCCEventBlock * e,bool unpack, bool forceToKeepFRdata)
  : DCCDataBlockPrototype(u,m,e,unpack), checkFeId_(false), forceToKeepFRdata_(forceToKeepFRdata) {
   
  expXtalTSamples_           = mapper_->numbXtalTSamples();
  numbDWInXtalBlock_         = (expXtalTSamples_-2)/4+1;
  unfilteredDataBlockLength_ = mapper_->getUnfilteredTowerBlockLength();
  xtalGains_                 = new short[expXtalTSamples_]; 
  
}


void DCCFEBlock::updateCollectors(){

  invalidBlockLengths_    = unpacker_->invalidBlockLengthsCollection();
  invalidTTIds_           = unpacker_->invalidTTIdsCollection();
  invalidZSXtalIds_       = unpacker_->invalidZSXtalIdsCollection();
}



int DCCFEBlock::unpack(uint64_t ** data, uint * dwToEnd, bool zs, uint expectedTowerID){
  
  zs_      = zs;  
  datap_   = data;
  data_    = *data;
  dwToEnd_ = dwToEnd;
  
  uint activeDCC = mapper_->getActiveSM();

  if( (*dwToEnd_)<1){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiTowerSize")
        <<"\n Unable to unpack Tower block for event "<<event_->l1A()<<" in fed "<<activeDCC
        <<"\n The end of event was reached "
        <<"\n(or, previously, pointers intended to navigate outside of FedBlock (based on block sizes), and were stopped by setting dwToEnd_ to zero)"    ;
      //TODO : add this to a dcc event size collection error?
    }
    return STOP_EVENT_UNPACKING;
  }
  
  lastStripId_     = 0;
  lastXtalId_      = 0;
  expTowerID_      = expectedTowerID;
  
  
  //Point to begin of block
  data_++;
  
  towerId_           = ( *data_ )                   & TOWER_ID_MASK;
  nTSamples_         = ( *data_>>TOWER_NSAMP_B  )   & TOWER_NSAMP_MASK; 
  bx_                = ( *data_>>TOWER_BX_B     )   & TOWER_BX_MASK;
  l1_                = ( *data_>>TOWER_L1_B     )   & TOWER_L1_MASK;
  blockLength_       = ( *data_>>TOWER_LENGTH_B )   & TOWER_LENGTH_MASK;
  
  event_->setFESyncNumbers(l1_,bx_, (short)(expTowerID_-1));

  //debugging
  //display(cout);


  
  ////////////////////////////////////////////////////
  // check that expected fe_id==fe_expected is on
  if( checkFeId_              &&
      expTowerID_ != towerId_ &&
      expTowerID_ <= mapper_->getNumChannelsInDcc(activeDCC) ){ // fe_id must be within range foreseen in the FED 
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiTowerId")
        <<"\n For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
        <<"\n Expected FE_id is "<<expTowerID_<<" while "<<towerId_<<" was found "
        <<"\n => Skipping to next FE block...";
     } 
   
    fillEcalElectronicsError(invalidTTIds_); 
    
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }

  //////////////////////////////////////////////////////////
  // check that expected fe_id==fe_expected is off
  else if( (!checkFeId_) && 
	   towerId_ > mapper_->getNumChannelsInDcc(activeDCC) ){ // fe_id must still be within range foreseen in the FED 
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiTowerId")
        <<"\n For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()<<" (there's no check fe_id==dcc_channel)"
        <<"\n the FE_id found: "<<towerId_<<" exceeds max number of FE foreseen in fed"
        <<"\n => Skipping to next FE block...";
    }
    
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }
  
  
  
  // Check synchronization
  if(sync_){

    uint dccBx = (event_->bx())&TCC_BX_MASK;
    uint dccL1 = (event_->l1A())&TCC_L1_MASK; 
    // accounting for counters starting from 0 in ECAL FE, while from 1 in CSM
    if( dccBx != bx_ || dccL1 != (l1_+1) ){
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("EcalRawToDigiNumTowerBlocks")
	  <<"\n Synchronization error for Tower Block "<<towerId_<<" in event "<<event_->l1A()
	  <<" with bx "<<event_->bx()<<" in fed "<<mapper_->getActiveDCC()
          <<"\n TCC local l1A is  "<<l1_<<" and local bx is "<<bx_
          <<"\n => Skipping to next tower block...";
       }
      //Note : add to error collection ?		 
      updateEventPointers();
      return SKIP_BLOCK_UNPACKING;
    }
  }



  // check number of samples
  if( nTSamples_ != expXtalTSamples_ ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiNumTowerBlocks")
        <<"EcalRawToDigi@SUB=DCCFEBlock::unpack"
        <<"\n Unable to unpack Tower Block "<<towerId_<<" for event L1A "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Number of time samples "<<nTSamples_<<" is not the same as expected ("<<expXtalTSamples_<<")"
        <<"\n => Skipping to next tower block...";
     } 
    //Note : add to error collection ?		 
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }

  
  xtalBlockSize_     = numbDWInXtalBlock_*8;
  blockSize_           = blockLength_*8;  
  
  if((*dwToEnd_)<blockLength_){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiNumTowerBlocks")
        <<"\n Unable to unpack Tower Block "<<towerId_<<" for event L1A "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<blockSize_<<" are needed!"
        <<"\n => Skipping to next fed block...";
    }
    //TODO : add to error collections
    return STOP_EVENT_UNPACKING;
  }


  if(!zs_ && !forceToKeepFRdata_){
	 
    if ( unfilteredDataBlockLength_ != blockLength_ ){
      if( ! DCCDataUnpacker::silentMode_ ){ 
        edm::LogWarning("EcalRawToDigiNumTowerBlocks")
          <<"\n For event L1A "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n Expected block size is "<<(unfilteredDataBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found"
          <<"\n => Skipping to next fed block...";
       }

      fillEcalElectronicsError(invalidBlockLengths_) ;

      //Safer approach...  - why pointers do not navigate in this case?
      return STOP_EVENT_UNPACKING;	  

      
    }

    
  }
  else if (!zs && forceToKeepFRdata_){

     if ( unfilteredDataBlockLength_ != blockLength_ ){
      if( ! DCCDataUnpacker::silentMode_ ){ 
        edm::LogWarning("EcalRawToDigiNumTowerBlocks")
          <<"\n For event L1A "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n Expected block size is "<<(unfilteredDataBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found"
          <<"\n => Keeps unpacking as the unpacker was forced to keep FR data (by configuration) ...";
       }

      fillEcalElectronicsError(invalidBlockLengths_) ;
     }

  }
  else if( blockLength_ > unfilteredDataBlockLength_ || (blockLength_-1) < numbDWInXtalBlock_ ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiNumTowerBlocks")
        <<"\n For event L1A "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
        <<"\n The tower "<<towerId_<<" has a wrong number of bytes : "<<(blockLength_*8)	   
        <<"\n => Skipping to next fed block...";
     }

     fillEcalElectronicsError(invalidBlockLengths_) ;


    //Safer approach... - why pointers do not navigate in this case?
    return STOP_EVENT_UNPACKING;
  }
  


  // If the HLT says to skip this tower we skip it...
  if( ! event_->getHLTChannel(towerId_) ){
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }
  /////////////////////////////////////////////////





  uint numbOfXtalBlocks = (blockLength_-1)/numbDWInXtalBlock_; 

  // get XTAL Data
  uint expStripID(0), expXtalID(0);
  //point to xtal data
  data_++;
  
  int statusUnpackXtal =0;

  for(uint numbXtal=1; numbXtal <= numbOfXtalBlocks && statusUnpackXtal!= SKIP_BLOCK_UNPACKING; numbXtal++){

    
    if(!zs_ && ! forceToKeepFRdata_){
      expStripID  = ( numbXtal-1)/5 + 1;	
      expXtalID   =  numbXtal - (expStripID-1)*5;
    }
    
    statusUnpackXtal = unpackXtalData(expStripID,expXtalID);
    if (statusUnpackXtal== SKIP_BLOCK_UNPACKING)
      {
        if( ! DCCDataUnpacker::silentMode_ ){
  	  edm::LogWarning("EcalRawToDigi")
	    <<"\n For event L1A "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
	    <<"\n The tower "<<towerId_<<" won't be unpacked further";
        }
      }

  }// end loop over xtals of given FE 

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


