#include "EventFilter/EcalRawToDigi/interface/DCCSRPBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"

DCCSRPBlock::DCCSRPBlock(
  DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack
) : DCCDataBlockPrototype(u,m,e,unpack)
{
 
  // Todo : include data integrity collections
  blockLength_    = SRP_BLOCKLENGTH;
  // Set SR flags to zero
  for(unsigned int i=0; i<SRP_NUMBFLAGS; i++){ srFlags_[i]=0; }

}


int DCCSRPBlock::unpack(uint64_t ** data, unsigned int * dwToEnd, unsigned int numbFlags ){    

  // Set SR flags to zero
  for(unsigned int i=0; i<SRP_NUMBFLAGS; i++){ srFlags_[i]=0; }
  

  expNumbSrFlags_ = numbFlags;
  error_          = false;  
  datap_          = data;
  data_           = *data;
  dwToEnd_        = dwToEnd;
  
  // Check SRP Length
  if( (*dwToEnd_) < blockLength_ ){
    if( ! DCCDataUnpacker::silentMode_ ){ 
      edm::LogWarning("IncorrectEvent")
        <<"\n Event "<<l1_
        <<"\n Unable to unpack SRP block for event "<<event_->l1A()<<" in fed <<"<<mapper_->getActiveDCC()
        <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<(blockLength_*8)<<" are needed!";
     }
    
    //Note : add to error collection 
    
    return STOP_EVENT_UNPACKING;
    
  }
  
  
  
  // Point to begin of block
  data_++;
  
  srpId_          = ( *data_ ) & SRP_ID_MASK; 
  bx_             = ( *data_>>SRP_BX_B     ) & SRP_BX_MASK;
  l1_             = ( *data_>>SRP_L1_B     ) & SRP_L1_MASK;
  nSRFlags_       = ( *data_>>SRP_NFLAGS_B ) & SRP_NFLAGS_MASK;
  
  event_->setSRPSyncNumbers(l1_,bx_);
 
  if( ! checkSrpIdAndNumbSRFlags() ){ 
    // SRP flags are required to check FE data 
	return  STOP_EVENT_UNPACKING; 
  }
	 
  // Check synchronization
  if(sync_){
    unsigned int dccL1 = ( event_->l1A() ) & SRP_BX_MASK;
    unsigned int dccBx = ( event_->bx()  ) & SRP_L1_MASK;
    if( dccBx != bx_ || dccL1 != l1_ ){
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("IncorrectEvent")
          <<"EcalRawToDigi@SUB=DCCSRPBlock::unpack"
          <<"\nSynchronization error for SRP block in event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
          <<"\n SRP local l1A is  "<<l1_<<" and local bx is "<<bx_;
      }
       //Note : add to error collection ?		 
       // SRP flags are required to check FE , better using synchronized data...
	   return STOP_EVENT_UNPACKING;
    }
  } 

  // initialize array, protecting in case of inconsistently formatted data
  for(int dccCh=0; dccCh<SRP_NUMBFLAGS; dccCh++) srFlags_[dccCh] =0;
  
  //display(cout); 
  addSRFlagToCollection();
  
  updateEventPointers();
  
  return true;
        
}



void DCCSRPBlock::display(std::ostream& o){

  o<<"\n Unpacked Info for SRP Block"
  <<"\n DW1 ============================="
  <<"\n SRP Id "<<srpId_
  <<"\n Numb Flags "<<nSRFlags_ 
  <<"\n Bx "<<bx_
  <<"\n L1 "<<l1_;
 
  for(unsigned int i=0; i<SRP_NUMBFLAGS; i++){ 
    o<<"\n SR flag "<<(i+1)<<" = "<<(srFlags_[i]); 
  } 
} 




