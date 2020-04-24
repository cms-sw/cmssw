
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalDCCHeaderRuntypeDecoder.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCMemBlock.h"


#include "EventFilter/EcalRawToDigi/interface/DCCEBEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEBTCCBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEBSRPBlock.h"
#include <sys/time.h>

#include <iomanip>
#include <sstream>


DCCEBEventBlock::DCCEBEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m , bool hU, bool srpU, bool tccU, bool feU , bool memU, bool forceToKeepFRdata) : 
  DCCEventBlock(u,m,hU,srpU,tccU,feU,memU,forceToKeepFRdata)
{

  //Builds a tower unpacker block
  towerBlock_ = new DCCTowerBlock(u,m,this,feUnpacking_, forceToKeepFRdata_); 
  
  //Builds a srp unpacker block
  srpBlock_   = new DCCEBSRPBlock(u,m,this,srpUnpacking_);
  
  //Builds a tcc unpacker block
  tccBlock_   = new DCCEBTCCBlock(u,m,this,tccUnpacking_);

  // This field is not used in EB
  mem_ = 0;    
  
 
}



void DCCEBEventBlock::unpack(const uint64_t * buffer, size_t numbBytes, unsigned int expFedId){
  
  reset();
 
  eventSize_ = numbBytes;        
  data_      = buffer;
  
  // First Header Word of fed block
  fedId_             = ((*data_)>>H_FEDID_B)   & H_FEDID_MASK;
  bx_                = ((*data_)>>H_BX_B   )   & H_BX_MASK;
  l1_                = ((*data_)>>H_L1_B   )   & H_L1_MASK;
  triggerType_       = ((*data_)>>H_TTYPE_B)   & H_TTYPE_MASK;
  
  // Check if fed id is the same as expected...
  if( fedId_ != expFedId  ){ 

  if( ! DCCDataUnpacker::silentMode_ ){  
    edm::LogWarning("IncorrectEvent")
      <<"\n For event L1A: "<<l1_
      <<"\n Expected FED id is: "<<expFedId<<" while current FED id is: "<<fedId_
      <<"\n => Skipping to next fed block...";
    }
  
  //TODO : add this to an error event collection
  
  return;
  } 
  
  // Check if this event is an empty event 
  if( eventSize_ == EMPTYEVENTSIZE ){ 
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectEvent")
        <<"\n Event L1A: "<<l1_<<" is empty for fed: "<<fedId_
        <<"\n => Skipping to next fed block...";
    }
    return;
    
  } 
  
  //Check if event size allows at least building the header
  else if( eventSize_ < HEADERSIZE ){    
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogError("IncorrectEvent")
        <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
        <<"\n Event size is "<<eventSize_<<" bytes while the minimum is "<<HEADERSIZE<<" bytes"
        <<"\n => Skipping to next fed block..."; 
     }
    
    //TODO : add this to a dcc size error collection  
    
    return;
    
  }
  
  //Second Header Word of fed block
  data_++;
  
  blockLength_   =   (*data_ )                 & H_EVLENGTH_MASK;
  dccErrors_     =   ((*data_)>>H_ERRORS_B)    & H_ERRORS_MASK;
  runNumber_     =   ((*data_)>>H_RNUMB_B )    & H_RNUMB_MASK;
  
  
  if( eventSize_ != blockLength_*8 ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogError("IncorrectEvent")
        <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
        <<"\n size is "<<eventSize_<<" bytes while "<<(blockLength_*8)<<" are set in the event header "
        <<"\n => Skipping to next fed block..."; 
      //TODO : add this to a dcc size error collection 
     }
    return;
    
  }  
  
  //Third Header Word  of fed block
  data_++;


  // bits 0.. 31 of the 3rd DCC header word
  runType_              = (*data_) & H_RTYPE_MASK;
  
  fov_                  = ((*data_)>>H_FOV_B) & H_FOV_MASK;

  // bits 32.. 47 of the 3rd DCC header word
  detailedTriggerType_ = ((*data_) >> H_DET_TTYPE_B) & H_DET_TTYPE_MASK;

  //Forth Header Word
  data_++;
  orbitCounter_        = ((*data_)>>H_ORBITCOUNTER_B)  & H_ORBITCOUNTER_MASK;
  sr_                  = ((*data_)>>H_SR_B)            & B_MASK;
  zs_                  = ((*data_)>>H_ZS_B)            & B_MASK;
  tzs_                 = ((*data_)>>H_TZS_B)           & B_MASK;
  srChStatus_          = ((*data_)>>H_SRCHSTATUS_B)    & H_CHSTATUS_MASK;
  

  bool ignoreSR(true);

  // getting TCC channel status bits
  tccChStatus_[0] = ((*data_)>>H_TCC1CHSTATUS_B)   & H_CHSTATUS_MASK; 
  tccChStatus_[1] = ((*data_)>>H_TCC2CHSTATUS_B)   & H_CHSTATUS_MASK;
  tccChStatus_[2] = ((*data_)>>H_TCC3CHSTATUS_B)   & H_CHSTATUS_MASK;
  tccChStatus_[3] = ((*data_)>>H_TCC4CHSTATUS_B)   & H_CHSTATUS_MASK;
    
  // FE  channel Status data
  int channel(0);
  for( int dw = 0; dw<5; dw++ ){
    data_++;
    for( int i = 0; i<14; i++, channel++){
      unsigned int shift = i*4; //each channel has 4 bits
      feChStatus_[channel] = ( (*data_)>>shift ) &  H_CHSTATUS_MASK ;
    }
  }
   
  // debugging
  //display(cout);
  
  // Update number of available dwords
  dwToEnd_ = blockLength_ - HEADERLENGTH ;
   
  int STATUS = unpackTCCBlocks();

  if(  STATUS != STOP_EVENT_UNPACKING && (feUnpacking_ || srpUnpacking_) ){
    
    //NMGA note : SR comes before TCC blocks 
    // Emmanuelle please change this in the digi to raw
  
    // Unpack SRP block
    if(srChStatus_ != CH_TIMEOUT &&  srChStatus_ != CH_DISABLED){
      STATUS = srpBlock_->unpack(&data_,&dwToEnd_,SRP_NUMBFLAGS);
      if ( STATUS == BLOCK_UNPACKED ){ ignoreSR = false; }
    }
    
  }





  // See number of FE channels that we need according to the trigger type //
  // TODO : WHEN IN LOCAL MODE WE SHOULD CHECK RUN TYPE                        
  unsigned int numbChannels(0);
  
  if(       triggerType_ == PHYSICTRIGGER      ){ numbChannels = 68; }
  else if ( triggerType_ == CALIBRATIONTRIGGER ){ numbChannels = 70; }
  else {
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogError("IncorrectEvent")
        <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
        <<"\n Event has an unsupported trigger type "<<triggerType_
        <<"\n => Skipping to next fed block..."; 
      //TODO : add this to a dcc trigger type error collection 
    }
    return;
  }
  
  // note: there is no a-priori check that number_active_channels_from_header
  //          equals number_channels_found_in_data.
  //          The checks are doing f.e. by f.e. only.
  
  if( feUnpacking_ || memUnpacking_ ){
    // pointer for the
    std::vector<short>::iterator it = feChStatus_.begin();
    
    // fields for tower recovery code
    unsigned int next_tower_id = 1000;
    const uint64_t* next_data = data_;
    unsigned int next_dwToEnd = dwToEnd_;
    
    // looping over FE channels, i.e. tower blocks
    for (unsigned int chNumber=1; chNumber<= numbChannels && STATUS!=STOP_EVENT_UNPACKING; chNumber++, it++ ){
      //for( unsigned int i=1; chNumber<= numbChannels; chNumber++, it++ ){                        

      const short chStatus(*it);
      
      // regular cases
      const bool regular = (chStatus == CH_DISABLED ||
                            chStatus == CH_SUPPRESS);
      
      // problematic cases
      const bool problematic = (chStatus == CH_TIMEOUT ||
                                chStatus == CH_HEADERERR ||
                                chStatus == CH_LINKERR ||
                                chStatus == CH_LENGTHERR ||
                                chStatus == CH_IFIFOFULL ||
                                chStatus == CH_L1AIFIFOFULL);
      
      // issuiung messages for problematic cases, even though handled by the DCC
      if (problematic) {
        if (! DCCDataUnpacker::silentMode_ ) {
          const int val = unpacker_->getCCUValue(fedId_, chNumber);
          const bool ttProblem = (val == 13) || (val == 14);
          if (! ttProblem) {
            edm::LogWarning("IncorrectBlock")
              << "Bad DCC channel status: " << chStatus
              << " (LV1 " << l1_ << " fed " << fedId_ << " tower " << chNumber << ")\n"
              << "  => DCC channel is not being unpacked";
          }
        }
      }
      
      // skip unpack in case of bad status
      if (regular || problematic) {
        continue;
      }
      
      // preserve data pointer
      const uint64_t* const prev_data = data_;
      const unsigned int prev_dwToEnd = dwToEnd_;
      
      // skip corrupted/problematic data block
      if (chNumber >= next_tower_id) {
        data_ = next_data;
        dwToEnd_ = next_dwToEnd;
        next_tower_id = 1000;
      }
      
      
      // Unpack Tower (Xtal Block)
      if (feUnpacking_ && chNumber <= 68) {
        
        //  in case of SR (data are 0 suppressed)
        if (sr_) {
          const bool applyZS =
            (fov_ == 0) ||      // backward compatibility with FOV = 0;
            ignoreSR ||
            (chStatus == CH_FORCEDZS1) ||
            ((srpBlock_->srFlag(chNumber) & SRP_SRVAL_MASK) != SRP_FULLREADOUT);
          
          STATUS = towerBlock_->unpack(&data_, &dwToEnd_, applyZS, chNumber);
          
              // If there is an action to suppress SR channel the associated channel status should be updated 
              // so we can remove this piece of code
              // if ( ( srpBlock_->srFlag(chNumber) & SRP_SRVAL_MASK) != SRP_NREAD ){
              //
              //  STATUS = towerBlock_->unpack(&data_,&dwToEnd_,applyZS,chNumber);
              //}
        }
        // no SR (possibly 0 suppression flags)
        else {
          // if tzs_ data are not really suppressed, even though zs flags are calculated
          if(tzs_){ zs_ = false;}
          STATUS = towerBlock_->unpack(&data_,&dwToEnd_,zs_,chNumber);
        }
      }
      
      
      // Unpack Mem blocks
      if (memUnpacking_ && chNumber > 68) {
          STATUS = memBlock_->unpack(&data_,&dwToEnd_,chNumber);
      }
      
      
      // corruption recovery
      if (STATUS == SKIP_BLOCK_UNPACKING) {
        data_ = prev_data;
        dwToEnd_ = prev_dwToEnd;
        
        next_tower_id = next_tower_search(chNumber);
        
        next_data = data_;
        next_dwToEnd = dwToEnd_;
        
        data_ = prev_data;
        dwToEnd_ = prev_dwToEnd;
      }
      
    }
    // closing loop over FE/TTblock channels
    
  }// check if we need to perform unpacking of FE or mem data
  
  
  if(headerUnpacking_) addHeaderToCollection();
  
  
}



 // Unpack TCC blocks
int DCCEBEventBlock::unpackTCCBlocks(){

    if(tccChStatus_[0] != CH_TIMEOUT && tccChStatus_[0] != CH_DISABLED)
      return tccBlock_->unpack(&data_,&dwToEnd_,0);
    else return BLOCK_UNPACKED;

}
