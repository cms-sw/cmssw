
#include "EventFilter/EcalRawToDigiDev/interface/DCCTCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"

DCCTCCBlock::DCCTCCBlock ( DCCDataUnpacker  * u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack) : 
DCCDataBlockPrototype(u,m,e,unpack){}

 
void DCCTCCBlock::unpack(uint64_t ** data, uint * dwToEnd){ 
 
  dwToEnd_    = dwToEnd;  
  datap_      = data;
  data_       = *data;
  
  if( (*dwToEnd_)<blockLength_ ){
 
      std::ostringstream output;
      output<<"EcalRawToDigi@SUB=DCCTCCBlock:unpack"
        <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in dcc "<<mapper_->getActiveDCC()
        <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available until the end of event while "<<(blockLength_*8)<<" are needed!"
        <<"\n => Skipping this event..."<<std::endl;
      //todo : add this to error colection
      throw ECALUnpackerException(output.str()); 
  }
  
  if(unpackInternalData_){ 
  
    //  Go to the begining of the tcc block
    data_++;
  
    tccId_    = ( *data_ )           & TCC_ID_MASK;
    bx_       = ( *data_>>TCC_BX_B ) & TCC_BX_MASK;
    l1_       = ( *data_>>TCC_L1_B ) & TCC_L1_MASK;
    nTTs_     = ( *data_>>TCC_TT_B ) & TCC_TT_MASK;
    nTSamples_= ( *data_>>TCC_TS_B ) & TCC_TS_MASK;

    checkTccIdAndNumbTTs();  
  
    // Check synchronization
    if(sync_){
      uint dccBx = (event_->bx())  & TCC_BX_MASK;
      uint dccL1 = (event_->l1A()) & TCC_L1_MASK;    
      if( dccBx != bx_ || dccL1 != l1_ ){
        std::ostringstream output;
        output<<"EcalRawToDigi@SUB=DCCTCCBlock::unpack"
        <<"\n Synchronization error for TCC block in event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in dcc <<"<<mapper_->getActiveDCC()
        <<"\n TCC local l1A is  "<<l1_<<" and local bx is "<<bx_
        <<"\n Skipping this event..."<<std::endl;
        //Note : add to error collection ?		 
        throw ECALUnpackerException(output.str());
      }
    }
    
    //check numb of samples
   /*  
    uint expTriggerTSamples(mapper_->numbTriggerTSamples());
    
    if( nTSamples_ != expTriggerTSamples ){
      ostringstream output;
      output<<"EcalRawToDigi@SUB=DCCTCCBlock::unpack"
        <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in dcc "<<mapper_->getActiveDCC()
        <<"\n Number of time samples is "<<nTSamples_<<" while "<<expTriggerTSamples<<" is expected"
        <<"\n => Skipping this event..."<<endl;
      //Note : add to error collection ?		 
      throw ECALUnpackerException(output.str());
    }
    */	 
  
    // debugging
    // display(cout);

    addTriggerPrimitivesToCollection();

  } 

  updateEventPointers();	
  
}



void DCCTCCBlock::display(std::ostream& o){

  o<<"\n Unpacked Info for DCC TCC Block"
   <<"\n DW1 ============================="
   <<"\n TCC Id "<<tccId_
   <<"\n Bx "<<bx_
   <<"\n L1 "<<l1_
   <<"\n Numb TT "<<nTTs_
   <<"\n Numb Samp "<<nTSamples_
   <<std::endl;  
} 
    
