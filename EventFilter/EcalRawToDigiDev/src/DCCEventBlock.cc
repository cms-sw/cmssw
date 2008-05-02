#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalDCCHeaderRuntypeDecoder.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCFEBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCMemBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCTCCBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCSRPBlock.h"
#include <sys/time.h>

#include <iomanip>
#include <sstream>

DCCEventBlock::DCCEventBlock( DCCDataUnpacker * u , EcalElectronicsMapper * m ,  bool hU, bool srpU, bool tccU, bool feU, bool memU) : 
  unpacker_(u), mapper_(m), headerUnpacking_(hU), srpUnpacking_(srpU), tccUnpacking_(tccU), feUnpacking_(feU),memUnpacking_(memU)
{
  
  // Build a Mem Unpacker Block
  memBlock_   = new DCCMemBlock(u,m,this);
 
  // setup and initialize ch status vectors
  for( int feChannel=1;  feChannel <= 70;  feChannel++) { feChStatus_.push_back(0);}
  for( int tccChannel=1; tccChannel <= 4 ; tccChannel++){ tccChStatus_.push_back(0);}
  
}



void DCCEventBlock::enableSyncChecks(){
   towerBlock_   ->enableSyncChecks();
   tccBlock_     ->enableSyncChecks();
   memBlock_     ->enableSyncChecks();
   srpBlock_     ->enableSyncChecks();
}



void DCCEventBlock::updateCollectors(){

  dccHeaders_  = unpacker_->dccHeadersCollection();

  memBlock_    ->updateCollectors(); 
  tccBlock_    ->updateCollectors();
  srpBlock_    ->updateCollectors();
  towerBlock_  ->updateCollectors();
  
}


void DCCEventBlock::unpack( uint64_t * buffer, uint numbBytes, uint expFedId){
  
  eventSize_ = numbBytes;	
  data_      = buffer;
  
  // First Header Word of fed block
  fedId_             = ((*data_)>>H_FEDID_B)   & H_FEDID_MASK;
  bx_                = ((*data_)>>H_BX_B   )   & H_BX_MASK;
  l1_                = ((*data_)>>H_L1_B   )   & H_L1_MASK;
  triggerType_       = ((*data_)>>H_TTYPE_B)   & H_TTYPE_MASK;
  
  // Check if fed id is the same as expected...
  if( fedId_ != expFedId  ){ 
    
  edm::LogWarning("EcalRawToDigiDev")
    <<"\n For event L1A: "<<l1_
    <<"\n Expected FED id is: "<<expFedId<<" while current FED id is: "<<fedId_
    <<"\n => Skipping to next fed block...";
  
  //TODO : add this to an error event collection
  
  return;
  } 
  
  // Check if this event is an empty event 
  if( eventSize_ == EMPTYEVENTSIZE ){ 
    
    edm::LogWarning("EcalRawToDigiDev")
      <<"\n Event L1A: "<<l1_<<" is empty for fed: "<<fedId_
      <<"\n => Skipping to next fed block...";
   
    return;
    
  } 
  
  //Check if event size allows at least building the header
  else if( eventSize_ < HEADERSIZE ){    
    
    edm::LogError("EcalRawToDigiDev")
      <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
      <<"\n Event size is "<<eventSize_<<" bytes while the minimum is "<<HEADERSIZE<<" bytes"
      <<"\n => Skipping to next fed block..."; 
    
    //TODO : add this to a dcc size error collection  
    
    return;
    
  }
  
  //Second Header Word of fed block
  data_++;
  
  blockLength_   =   (*data_ )                 & H_EVLENGTH_MASK;
  dccErrors_     =   ((*data_)>>H_ERRORS_B)    & H_ERRORS_MASK;
  runNumber_     =   ((*data_)>>H_RNUMB_B )    & H_RNUMB_MASK;
  
  
  if( eventSize_ != blockLength_*8 ){
    
    edm::LogError("EcalRawToDigiDev")
      <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
      <<"\n size is "<<eventSize_<<" bytes while "<<(blockLength_*8)<<" are set in the event header "
      <<"\n => Skipping to next fed block..."; 
    //TODO : add this to a dcc size error collection 
    return;
    
  }  
  
  //Third Header Word  of fed block
  data_++;

  // bits 0.. 31 of the 3rd DCC header word
  runType_              = (*data_) & H_RTYPE_MASK;

  // bits 32.. 47 of the 3rd DCC header word
  detailedTriggerType_ = ((*data_) >> H_DET_TTYPE_B) & H_DET_TTYPE_MASK;

  //Forth Header Word
  data_++;
  orbitCounter_        = ((*data_)>>H_ORBITCOUNTER_B)  & H_ORBITCOUNTER_MASK;
  sr_                  = ((*data_)>>H_SR_B)            & B_MASK;
  zs_                  = ((*data_)>>H_ZS_B)            & B_MASK;
  tzs_                 = ((*data_)>>H_TZS_B)           & B_MASK;
  srChStatus_          = ((*data_)>>H_SRCHSTATUS_B)    & H_CHSTATUS_MASK;
  
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
      uint shift = i*4; //each channel has 4 bits
      feChStatus_[channel] = ( (*data_)>>shift ) &  H_CHSTATUS_MASK ;
    }
  }
   
  // debugging
  //display(cout);
  
  if(headerUnpacking_) addHeaderToCollection();
  
  // pointer for the 
  std::vector<short>::iterator it;
  
  // Update number of available dwords
  dwToEnd_ = blockLength_ - HEADERLENGTH ;
   
  int STATUS = unpackTCCBlocks();

  if(  STATUS != STOP_EVENT_UNPACKING && feUnpacking_ || srpUnpacking_ ){
    
    //NMGA note : SR comes before TCC blocks 
    // Emmanuelle please change this in the digi to raw
  
    // Unpack SRP block
    if(srChStatus_ != CH_TIMEOUT &&  srChStatus_ != CH_DISABLED){
      STATUS = srpBlock_->unpack(&data_,&dwToEnd_);
    }
  }

  // See number of FE channels that we need according to the trigger type //
  // TODO : WHEN IN LOCAL MODE WE SHOULD CHECK RUN TYPE			
  uint numbChannels(0);
  
  if(       triggerType_ == PHYSICTRIGGER      ){ numbChannels = 68; }
  else if ( triggerType_ == CALIBRATIONTRIGGER ){ numbChannels = 70; }
  else {
    edm::LogError("EcalRawToDigiDev")
      <<"\n Event L1A: "<<l1_<<" in fed: "<< fedId_
      <<"\n Event has an unsupported trigger type "<<triggerType_
      <<"\n => Skipping to next fed block..."; 
    //TODO : add this to a dcc trigger type error collection 
    return;
  }
  
  // note: there is no a-priori check that number_active_channels_from_header
  //          equals number_channels_found_in_data.
  //          The checks are doing f.e. by f.e. only.
  
  if( feUnpacking_ || memUnpacking_ ){ 	     					
    it = feChStatus_.begin();
    
    // looping over FE channels, i.e. tower blocks
    for( uint chNumber=1; chNumber<= numbChannels && STATUS!=STOP_EVENT_UNPACKING; chNumber++, it++ ){			
      //for( uint i=1; chNumber<= numbChannels; chNumber++, it++ ){			

      short  chStatus(*it);
      
      // not issuiung messages for regular cases
      if(chStatus == CH_DISABLED ||
	 chStatus == CH_SUPPRESS) 
	{continue;}
      
      // issuiung messages for problematic cases, even though handled by the DCC
      else if( chStatus == CH_TIMEOUT || chStatus == CH_HEADERERR || chStatus == CH_LINKERR )
	{
	  edm::LogWarning("EcalRawToDigiDev") << "In fed: " << fedId_ << " at LV1: " << l1_
					      << " the DCC channel: " << chNumber 
					      << " has channel status: " << chStatus 
					      << " and is not being unpacked";
	  continue;
	}
      
      
      // Unpack Tower (Xtal Block) in case of SR (data are 0 suppressed)
      if(feUnpacking_ && sr_ && chNumber<=68)
	{
	  if ( srpBlock_->srFlag(chNumber) != SRP_NREAD ){
	    STATUS = towerBlock_->unpack(&data_,&dwToEnd_,true,chNumber);
	  }
	}
      
      
      
      // Unpack Tower (Xtal Block) for no SR (possibly 0 suppression flags)
      else if (feUnpacking_ && chNumber<=68)
	{
	  // if tzs_ data are not really suppressed, even though zs flags are calculated
	  if(tzs_){ zs_ = false;}
	  STATUS = towerBlock_->unpack(&data_,&dwToEnd_,zs_,chNumber);
	}
      
      
      // Unpack Mem blocks
      if(memUnpacking_	&& chNumber>68 )
	{
	  STATUS = memBlock_->unpack(&data_,&dwToEnd_,chNumber);
	}
      
    }
    // closing loop over FE/TTblock channels
    
  }// check if we need to perform unpacking of FE or mem data
  
}




void DCCEventBlock::addHeaderToCollection(){
  
  
  EcalDCCHeaderBlock theDCCheader;

  // container for fed_id (601-654 for ECAL) 
  theDCCheader.setFedId(fedId_);
  
  
  // this needs to be migrated to the ECAL mapping package

  // dccId is number internal to ECAL running 1.. 54.
  // convention is that dccId = (fed_id - 600)
  int dccId = mapper_->getActiveSM();
  // DCCHeaders follow  the same convenction
  theDCCheader.setId(dccId);
  

  theDCCheader.setRunNumber(runNumber_);  
  theDCCheader.setBasicTriggerType(triggerType_);
  theDCCheader.setLV1(l1_);
  theDCCheader.setBX(bx_);
  theDCCheader.setOrbit(orbitCounter_);
  theDCCheader.setErrors(dccErrors_);
  theDCCheader.setSelectiveReadout(sr_);
  theDCCheader.setZeroSuppression(zs_);
  theDCCheader.setTestZeroSuppression(tzs_);
  theDCCheader.setSrpStatus(srChStatus_);
  theDCCheader.setTccStatus(tccChStatus_);
  theDCCheader.setFEStatus(feChStatus_);

  EcalDCCHeaderRuntypeDecoder theRuntypeDecoder;
  uint DCCruntype              = runType_;
  uint DCCdetTriggerType = detailedTriggerType_;
  theRuntypeDecoder.Decode(triggerType_, DCCdetTriggerType , DCCruntype, &theDCCheader);

  // Add Header to collection 
  (*dccHeaders_)->push_back(theDCCheader);
   
}

void DCCEventBlock::display(std::ostream& o){
  o<<"\n Unpacked Info for DCC Event Class"
   <<"\n DW1 ============================="
   <<"\n Fed Id "<<fedId_
   <<"\n Bx "<<bx_
   <<"\n L1 "<<l1_
   <<"\n Trigger Type "<<triggerType_
   <<"\n DW2 ============================="	
   <<"\n Length "<<blockLength_
   <<"\n Dcc errors "<<dccErrors_
   <<"\n Run number "<<runNumber_
   <<"\n DW3 ============================="
   <<"\n SR "<<sr_
   <<"\n ZS "<<zs_
   <<"\n TZS "<<tzs_
   <<"\n SRStatus "<<srChStatus_;
	
  std::vector<short>::iterator it;
  int i(0),k(0);
  for(it = tccChStatus_.begin(); it!=tccChStatus_.end();it++,i++){
    o<<"\n TCCStatus#"<<i<<" "<<(*it);
  } 
  
  i=0;
  for(it = feChStatus_.begin();it!=feChStatus_.end();it++ ,i++){
    if(!(i%14)){ o<<"\n DW"<<(k+3)<<" ============================="; k++; }
    o<<"\n FEStatus#"<<i<<" "<<(*it);   	
  }

  o<<"\n";  
} 
    

DCCEventBlock::~DCCEventBlock(){
  if(towerBlock_){ delete towerBlock_; } 
  if(tccBlock_)  { delete tccBlock_;   }
  if(memBlock_)  { delete memBlock_;   }
  if(srpBlock_)  { delete srpBlock_;   }
}

