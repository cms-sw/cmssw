

#include "EventFilter/EcalRawToDigiDev/interface/DCCEBTCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

DCCEBTCCBlock::DCCEBTCCBlock ( DCCDataUnpacker  * u,  EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack) 
: DCCTCCBlock(u,m,e,unpack)
{
  blockLength_ = mapper_->getEBTCCBlockLength();
  expNumbTTs_  = TCC_EB_NUMBTTS;
}

void DCCEBTCCBlock::updateCollectors(){
  tps_ = unpacker_->ebTpsCollection();
}

void DCCEBTCCBlock::checkTccIdAndNumbTTs(){
    
  expTccId_ = mapper_->getActiveSM()+TCCID_SMID_SHIFT_EB;

  if( tccId_ != expTccId_ ){
    std::ostringstream output;
    output<<"EcalRawToDigi@SUB=DCCTCCBlock::unpack"
     <<"\n Error on event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in dcc "<<mapper_->getActiveDCC()
     <<"\n TCC id is "<<tccId_<<" while expected is "<<expTccId_
     <<"\n => Skipping this event...";
    //Note : add to error collection ?		 
    throw ECALUnpackerException(output.str());
  }
  
  //Check number of TT Flags
  if( nTTs_ != expNumbTTs_ ){
    std::ostringstream output;
    output<<"EcalRawToDigi@SUB=DCCTCCBlock::unpack"
     <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in dcc "<<mapper_->getActiveDCC()
     <<"\n Number of TTs "<<nTTs_<<" is different from expected "<<expNumbTTs_
     <<"\n => Skiping this event...";
     //Note : add to error collection ?		 
    throw ECALUnpackerException(output.str());
  }  

}


void DCCEBTCCBlock::addTriggerPrimitivesToCollection(){

  //point to trigger data
  data_++;

  uint16_t * tccP_= reinterpret_cast< uint16_t * >(data_);
 

  for( uint i = 1; i <= expNumbTTs_; i++){
   
    pTP_ =  mapper_->getTPPointer(tccId_,i);
	 
    for(uint ns = 0; ns<nTSamples_;ns++,tccP_++){
      
      pTP_->setSampleValue(ns, (*tccP_));
      (*tps_)->push_back(*pTP_);
        	  
    }
  }


}


