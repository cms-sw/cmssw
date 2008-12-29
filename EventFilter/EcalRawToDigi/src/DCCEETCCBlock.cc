#include "EventFilter/EcalRawToDigi/interface/DCCEETCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"


DCCEETCCBlock::DCCEETCCBlock ( DCCDataUnpacker  * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack) : 
DCCTCCBlock(u,m,e,unpack)
{
  blockLength_ = mapper_->getEETCCBlockLength();
}

void DCCEETCCBlock::updateCollectors(){
  tps_ = unpacker_->ecalTpsCollection();
}




void DCCEETCCBlock::addTriggerPrimitivesToCollection(){

  //point to trigger data
  data_++;

  uint16_t * tccP_= reinterpret_cast< uint16_t * >(data_);
  
  for( uint i = 1; i <= expNumbTTs_; i++){

    pTP_ = mapper_->getTPPointer(tccId_,i);

  if(pTP_){

    pTP_ =  mapper_->getTPPointer(tccId_,i);

    for(uint ns = 0; ns<nTSamples_;ns++,tccP_++){
      pTP_->setSample(ns, *tccP_ );
      (*tps_)->push_back(*pTP_);
     }
    }else{ break; } //if invalid we dont have more tts
	 
  }

}


bool DCCEETCCBlock::checkTccIdAndNumbTTs(){
	
	
  bool tccFound(false);
  int  activeDCC =  mapper_->getActiveSM();
  std::vector<uint> * m = mapper_->getTccs(activeDCC);
  std::vector<uint>::iterator it;
  for(it= m->begin();it!=m->end();it++){
    if((*it) == tccId_){ 
      tccFound=true;
      expNumbTTs_= 28; //separate from inner and outer tcc 
	
	/*
	  For inner TCCs you expect 28 TTs (4 in phi, 7 in eta).
	  For outer TCCs you expect 16 TTs (4 in phi, 4 in eta).
        */
 
	
	/*
	 to implement : map tccid-> number of tts
	 if( nTTs_ != xxx ){
        ostringstream output;
        output<<"EcalRawToDigi@SUB=DCCTCCBlock::unpack"
          <<"\n Unable to unpack TCC block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
          <<"\n Number of TTs "<<nTTs_<<" while "<<xxx<<" are expected";
        //Note : add to error collection ?		 
         throw ECALUnpackerException(output.str());
       }
	 */  
       break;
     }
	
   }
	
  if(!tccFound){

    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("EcalRawToDigiDevTCC") 
        <<"\n Error on event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
        <<"\n TCC id "<<tccId_<<" is not valid for this dcc "
        <<"\n => Skipping to next fed block...";
       //todo : add to error collection   
     }
  }

 return tccFound;

}


