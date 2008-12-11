#include "EventFilter/EcalRawToDigi/interface/DCCTCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"

DCCTCCBlock::DCCTCCBlock ( DCCDataUnpacker  * u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack) : 
DCCDataBlockPrototype(u,m,e,unpack){}

 
int DCCTCCBlock::unpack(uint64_t ** data, uint * dwToEnd, short tccChId){ 
 
  dwToEnd_    = dwToEnd;  
  datap_      = data;
  data_       = *data;
  
  // Need at least 1 dw to findout if pseudo-strips readout is enabled
  if(*dwToEnd == 1){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiTCC")
        <<"EcalRawToDigi@SUB=DCCTCCBlock:unpack"
        <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Only 8 bytes are available until the end of event ..."
        <<"\n => Skipping to next fed block...";
     }
    
    //todo : add this to error colection
    
    return STOP_EVENT_UNPACKING;
  }

  blockLength_ = getLength();
  
  
  if( (*dwToEnd_)<blockLength_ ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiTCC")
        <<"EcalRawToDigi@SUB=DCCTCCBlock:unpack"
        <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available until the end of event while "<<(blockLength_*8)<<" are needed!"
        <<"\n => Skipping to next fed block...";
     }
    
    //todo : add this to error colection
    
    return STOP_EVENT_UNPACKING;
  }
  
  
  
  if(unpackInternalData_){ 
  
    //  Go to the begining of the tcc block
    data_++;
  
    tccId_    = ( *data_ )           & TCC_ID_MASK;
    ps_       = ( *data_>>TCC_PS_B ) & B_MASK;      
    bx_       = ( *data_>>TCC_BX_B ) & TCC_BX_MASK;
    l1_       = ( *data_>>TCC_L1_B ) & TCC_L1_MASK;
    nTTs_     = ( *data_>>TCC_TT_B ) & TCC_TT_MASK;
    nTSamples_= ( *data_>>TCC_TS_B ) & TCC_TS_MASK;
	
    event_->setTCCSyncNumbers(l1_,bx_,tccChId);

    if ( ! checkTccIdAndNumbTTs() ){
	  updateEventPointers();
	  return SKIP_BLOCK_UNPACKING;
	}  
  
    // Check synchronization
    if(sync_){
      uint dccBx = (event_->bx())  & TCC_BX_MASK;
      uint dccL1 = (event_->l1A()) & TCC_L1_MASK;    
      if( dccBx != bx_ || dccL1 != l1_ ){
        if( ! DCCDataUnpacker::silentMode_ ){
          edm::LogWarning("EcalRawToDigiTCC")
  	    <<"\n Synchronization error for TCC block in event "<<event_->l1A()
	    <<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
	    <<"\n TCC local l1A is  "<<l1_<<" and local bx is "<<bx_
	    <<"\n TCC block skipped ...";
        }
	
	//Note : add to error collection ?	
	updateEventPointers();
	return SKIP_BLOCK_UNPACKING;
        
      }
    }
    
    //check numb of samples
   /*  
    uint expTriggerTSamples(mapper_->numbTriggerTSamples());
    
    if( nTSamples_ != expTriggerTSamples ){
      edm::LogWarning("EcalRawToDigiTCC")
        <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Number of time samples is "<<nTSamples_<<" while "<<expTriggerTSamples<<" is expected"
        <<"\n TCC block skipped..."<<endl;
		
       //Note : add to error collection ?
	   updateEventPointers();		 
      return SKIP_BLOCK_UNPACKING;
    }
    */	 
  
    // debugging
    // display(cout);

    addTriggerPrimitivesToCollection();

  } 

  updateEventPointers();
  return BLOCK_UNPACKED;	
  
}



void DCCTCCBlock::display(std::ostream& o){

  o<<"\n Unpacked Info for DCC TCC Block"
   <<"\n DW1 ============================="
   <<"\n TCC Id "<<tccId_
   <<"\n Bx "<<bx_
   <<"\n L1 "<<l1_
   <<"\n Numb TT "<<nTTs_
   <<"\n Numb Samp "<<nTSamples_;  
} 
    
