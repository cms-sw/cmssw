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


int DCCSRPBlock::unpack(const uint64_t ** data, unsigned int * dwToEnd, unsigned int numbFlags ){    

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
    //// SRP flags are required to check FE data 
    //return  STOP_EVENT_UNPACKING;
    updateEventPointers();
    return SKIP_BLOCK_UNPACKING;
  }
	 
  // Check synchronization
  if(sync_){
    const unsigned int dccL1 = ( event_->l1A() ) & SRP_L1_MASK;
    const unsigned int dccBx = ( event_->bx()  ) & SRP_BX_MASK;
    const unsigned int fov   = ( event_->fov() ) & H_FOV_MASK;
    
    if (! isSynced(dccBx, bx_, dccL1, l1_, TCC_SRP, fov)) {
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("IncorrectEvent")
          << "Synchronization error for SRP block"
          << " (L1A " << event_->l1A() << " bx " << event_->bx() << " fed " << mapper_->getActiveDCC() << ")\n"
          << "  dccBx = " << dccBx << " bx_ = " << bx_ << " dccL1 = " << dccL1 << " l1_ = " << l1_ << "\n"
          << "  => Stop event unpacking";
      }
       //Note : add to error collection ?		 
       //// SRP flags are required to check FE , better using synchronized data...
      //	   return STOP_EVENT_UNPACKING;
      updateEventPointers();
      return SKIP_BLOCK_UNPACKING;
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




