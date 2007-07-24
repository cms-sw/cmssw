
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
 
  // setup ch status 
  for( int feChannel=1;  feChannel <= 70;  feChannel++) { feChStatus_.push_back(0);}
  for( int tccChannel=1; tccChannel <= 4 ; tccChannel++){ tccChStatus_.push_back(0);}
  
}



void DCCEventBlock::enableSyncChecks(){
   towerBlock_->enableSyncChecks();
   tccBlock_->enableSyncChecks();
   memBlock_->enableSyncChecks();
   srpBlock_->enableSyncChecks();
}



void DCCEventBlock::updateCollectors(){

  dccHeaders_  = unpacker_->dccHeadersCollection();

  memBlock_->updateCollectors(); 
  tccBlock_->updateCollectors();
  srpBlock_->updateCollectors();
  towerBlock_->updateCollectors();
  
}


void DCCEventBlock::unpack( uint64_t * buffer, uint numbBytes, uint expFedId){
  
  eventSize_ = numbBytes;	
  data_      = buffer;
  
  // First Header Word
  fedId_       = ((*data_)>>H_FEDID_B) & H_FEDID_MASK;
  bx_          = ((*data_)>>H_BX_B   ) & H_BX_MASK;                        
  l1_          = ((*data_)>>H_L1_B   ) & H_L1_MASK;                          
  triggerType_ = ((*data_)>>H_TTYPE_B) & H_TTYPE_MASK; 
  
  // Check if fed id is the same as expected...
  if( fedId_ != expFedId  ){ 
    
  edm::LogWarning("EcalRawToDigiDev")
    <<"\n For event "<<l1_
    <<"\n Expected FED id is "<<expFedId<<" while current FED id is "<<fedId_
    <<"\n => Skipping this event...";

    //TODO : add this to an error event collection

	return;
  } 
  
  // Check if this event is an empty event 
  if( eventSize_ == EMPTYEVENTSIZE ){ 
  
    edm::LogWarning("EcalRawToDigiDev")
      <<"\n Event "<<l1_<<" is empty for dcc "<<fedId_
      <<"\n => Skipping this event...";
    
	//TODO : add this to a dcc empty event collection 	 
    
	return;
	
  } 

  //Check if event size allows at least building the header
  else if( eventSize_ < HEADERSIZE ){    
    
	edm::LogWarning("EcalRawToDigiDev")
      <<"\n Event "<<l1_<<" in dcc "<< fedId_
      <<"\n Event size is "<<eventSize_<<" bytes while the minimum is "<<HEADERSIZE<<" bytes"
      <<"\n => Skipping this event..."; 

    //TODO : add this to a dcc size error collection  

	return;
  
  }
  
  //Second Header Word
  data_++;
	 
  blockLength_  =  (*data_ )              & H_EVLENGTH_MASK;
  dccErrors_    =  ((*data_)>>H_ERRORS_B) & H_ERRORS_MASK  ;
  runNumber_    =  ((*data_)>>H_RNUMB_B ) & H_RNUMB_MASK   ;
   
  
  if( eventSize_ != blockLength_*8 ){
  
    edm::LogWarning("EcalRawToDigiDev")
      <<"\n Event "<<l1_<<" in dcc "<< fedId_
      <<"\n Event size is "<<eventSize_<<" bytes while "<<(blockLength_*8)<<" are set in the event header "
      <<"\n => Skipping this event ...";
    //TODO : add this to a dcc size error collection 
	return;
	 
  }  
  
  
  //Third Header Word
  data_++;

  // bits 0.. 31 of the 3rd DCC header word
  runType_                    = (*data_) & H_RTYPE_MASK;

  // bits 32.. 47 of the 3rd DCC header word
  detailedTriggerType_ = ((*data_) >> H_DET_TTYPE_B) & H_DET_TTYPE_MASK;

  //Forth Header Word
  data_++;
  sr_           = ((*data_)>>H_SR_B)  & B_MASK;
  zs_           = ((*data_)>>H_ZS_B)  & B_MASK;
  tzs_          = ((*data_)>>H_TZS_B) & B_MASK;
  srChStatus_   = ((*data_)>>H_SRCHSTATUS_B) & H_CHSTATUS_MASK;
  
  tccChStatus_[0] = ((*data_)>>H_TCC1CHSTATUS_B) & H_CHSTATUS_MASK; 
  tccChStatus_[1] = ((*data_)>>H_TCC2CHSTATUS_B) & H_CHSTATUS_MASK;
  tccChStatus_[2] = ((*data_)>>H_TCC3CHSTATUS_B) & H_CHSTATUS_MASK;
  tccChStatus_[3] = ((*data_)>>H_TCC4CHSTATUS_B) & H_CHSTATUS_MASK;
    
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
     edm::LogWarning("EcalRawToDigiDev")
     <<"\n Event "<<l1_<<" in dcc "<< fedId_
     <<"\n Event has an unsupported trigger type "<<triggerType_
     <<"\n => Skipping this event "; 
     
	 //TODO : add this to a dcc trigger type error collection 
     
	 return;
	 
  }  
  
  if( feUnpacking_ || memUnpacking_ ){ 	     					
    it = feChStatus_.begin();
    for( uint i=1; i<= numbChannels && STATUS!=STOP_EVENT_UNPACKING; i++, it++ ){			

      short  chStatus(*it);
    
      // Unpack Tower and Xtal Blocks
      if(sr_ && chStatus != CH_TIMEOUT && chStatus != CH_DISABLED && chStatus != CH_SUPPRESS && i<=68){
      
        if (feUnpacking_ && srpBlock_->srFlag(i) != SRP_NREAD ){
          STATUS = towerBlock_->unpack(&data_,&dwToEnd_,true,i);
        }
      
      }else if (feUnpacking_ && chStatus != CH_TIMEOUT && chStatus != CH_DISABLED && chStatus != CH_SUPPRESS && i<=68){
	    // if tzs_ data are not really suppressed, even though zs flags are calculated
        if(tzs_){ zs_ = false;}
        STATUS = towerBlock_->unpack(&data_,&dwToEnd_,zs_,i);
      }		 
  
      // Unpack Mem blocks
      if(memUnpacking_&& i>68 && chStatus != CH_TIMEOUT && chStatus != CH_DISABLED){
        STATUS = memBlock_->unpack(&data_,&dwToEnd_,i);
      }
		       				
    }// closing loop of channels
  
  }// check if we need to perform unpacking of fe or monitoring events

} 




void DCCEventBlock::addHeaderToCollection(){
  
  
  EcalDCCHeaderBlock theDCCheader;

  // container for fed_id (601-645 for ECAL) 
  theDCCheader.setFedId(fedId_);
  
  
  // this needs to be migrated to the ECAL mapping package

  // dccId is number internal to ECAL running 1.. 54.
  // convention is that dccId = (fed_id - 600)
  int dccId = mapper_->getActiveSM();

  // deriving ism starting from dccId
  int ism(0);
  if        (9< dccId && dccId < 28){
    ism  = dccId-9+18;}

  else if (27 < dccId && dccId< 46){
    ism  = dccId-9-18;}

  else
    {ism = -999;}
  
  theDCCheader.setId(ism);
  

  theDCCheader.setRunNumber(runNumber_);  
  theDCCheader.setBasicTriggerType(triggerType_);
  theDCCheader.setLV1(l1_);
  theDCCheader.setBX(bx_);
  theDCCheader.setErrors(dccErrors_);
  theDCCheader.setSelectiveReadout(sr_);
  theDCCheader.setZeroSuppression(zs_);
  theDCCheader.setTestZeroSuppression(tzs_);
  theDCCheader.setSrpStatus(srChStatus_);
  theDCCheader.setTccStatus(tccChStatus_);
  theDCCheader.setTriggerTowerStatus(feChStatus_);
  
  // The Run type
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

