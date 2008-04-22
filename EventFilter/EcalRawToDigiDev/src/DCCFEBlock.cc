#include "EventFilter/EcalRawToDigiDev/interface/DCCFEBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
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
   edm::LogWarning("EcalRawToDigiDevTowerSize")
      <<"\n Unable to unpack Tower block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
      <<"\n The end of event was reached "
      <<"\n(or, previously, pointers intended to navigate outside of FedBlock (based on block sizes), and were stopped by setting dwToEnd_ to zero)"    ;
    //TODO : add this to a dcc event size collection error?
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

  //debugging
  //display(cout);


  uint activeDCC = mapper_->getActiveSM();
  
  //check expected trigger tower id
  if( expTowerID_ != towerId_ &&
      expTowerID_ <= mapper_->getNumChannelsInDcc(activeDCC) ){
    
    edm::LogWarning("EcalRawToDigiDevTowerId")
      <<"\n For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
      <<"\n Expected trigger tower is "<<expTowerID_<<" while "<<towerId_<<" was found "
      <<"\n => Skipping to next tower block...";

    // in case of EB, FE is one-to-one with TT
    // use those EcalElectronicsId for simplicity
    if(NUMB_SM_EB_MIN_MIN<=activeDCC && activeDCC<=NUMB_SM_EB_PLU_MAX){
      EcalElectronicsId  *  eleTp = mapper_->getTTEleIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
      (*invalidTTIds_)->push_back(*eleTp);
    }// EE
    else if ( (NUMB_SM_EE_MIN_MIN <=activeDCC && activeDCC<=NUMB_SM_EE_MIN_MAX) ||
	      (NUMB_SM_EE_PLU_MIN <=activeDCC && activeDCC<=NUMB_SM_EE_PLU_MAX) )
      {
	EcalElectronicsId * scEleId = mapper_->getSCElectronicsPointer(activeDCC, expTowerID_);
	(*invalidTTIds_)->push_back(*scEleId);
      }
    else
      {
	edm::LogWarning("EcalRawToDigiDevChId")
	  <<"\n For event "<<event_->l1A()<<" there's fed: "<< mapper_->getActiveDCC()
	  <<" activeDcc: "<<mapper_->getActiveSM()
	  <<" but that activeDcc is not valid.";
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
      edm::LogWarning("EcalRawToDigiDevNumTowerBlocks")
	<<"\n Synchronization error for Tower Block "<<towerId_<<" in event "<<event_->l1A()
	<<" with bx "<<event_->bx()<<" in fed "<<mapper_->getActiveDCC()
       <<"\n TCC local l1A is  "<<l1_<<" and local bx is "<<bx_
      <<"\n => Skipping to next tower block...";
      //Note : add to error collection ?		 
      updateEventPointers();
      return SKIP_BLOCK_UNPACKING;
    }
  }



  // check number of samples
  if( nTSamples_ != expXtalTSamples_ ){

    edm::LogWarning("EcalRawToDigiDevNumTowerBlocks")
      <<"EcalRawToDigi@SUB=DCCFEBlock::unpack"
      <<"\n Unable to unpack Tower Block "<<towerId_<<" for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
      <<"\n Number of time samples "<<nTSamples_<<" is not the same as expected ("<<expXtalTSamples_<<")"
      <<"\n => Skipping to next tower block...";
    //Note : add to error collection ?		 
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }

  
  xtalBlockSize_     = numbDWInXtalBlock_*8;
  blockSize_           = blockLength_*8;  
  
  if((*dwToEnd_)<blockLength_){
    
    edm::LogWarning("EcalRawToDigiDevNumTowerBlocks")
      <<"\n Unable to unpack Tower Block "<<towerId_<<" for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
      <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<blockSize_<<" are needed!"
      <<"\n => Skipping to next fed block...";
    //TODO : add to error collections
    return STOP_EVENT_UNPACKING;
  }


  if(!zs_){
	 
    if ( unfilteredDataBlockLength_ != blockLength_ ){
      
      edm::LogWarning("EcalRawToDigiDevNumTowerBlocks")
        <<"\n For event "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
        <<"\n Expected block size is "<<(unfilteredDataBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found"
        <<"\n => Skipping to next fed block...";

      EcalElectronicsId  *  eleTp = mapper_->getTTEleIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
      (*invalidBlockLengths_)->push_back(*eleTp);

      //Safer approach...  - why pointers do not navigate in this case?
      return STOP_EVENT_UNPACKING;	  
    }

    
  }else if( blockLength_ > unfilteredDataBlockLength_ || (blockLength_-1) < numbDWInXtalBlock_ ){
    
    edm::LogWarning("EcalRawToDigiDevNumTowerBlocks")
      <<"\n For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
      <<"\n The tower "<<towerId_<<" has a wrong number of bytes : "<<(blockLength_*8)	   
      <<"\n => Skipping to next fed block...";

    EcalElectronicsId  *  eleTp = mapper_->getTTEleIdPointer(mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB,expTowerID_);
    (*invalidBlockLengths_)->push_back(*eleTp);

    //Safer approach... - why pointers do not navigate in this case?
    return STOP_EVENT_UNPACKING;
  }
  


  uint numbOfXtalBlocks = (blockLength_-1)/numbDWInXtalBlock_; 

  // get XTAL Data
  uint expStripID(0), expXtalID(0);
  //point to xtal data
  data_++;
  
  int statusUnpackXtal =0;

  for(uint numbXtal=1; numbXtal <= numbOfXtalBlocks && statusUnpackXtal!= SKIP_BLOCK_UNPACKING; numbXtal++){

    // If zs is disabled we know the expected strip and xtal ids
    // Note : this is valid for the EB how about the EE ? -> retieve expected index from mapper
    
    if(!zs_){
      expStripID  = ( numbXtal-1)/5 + 1;	
      expXtalID   =  numbXtal - (expStripID-1)*5;
    }
    
    statusUnpackXtal = unpackXtalData(expStripID,expXtalID);
    if (statusUnpackXtal== SKIP_BLOCK_UNPACKING)
      {
	edm::LogWarning("EcalRawToDigiDev")
	  <<"\n For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC()
	  <<"\n The tower "<<towerId_<<" won't be unpacked further";
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


