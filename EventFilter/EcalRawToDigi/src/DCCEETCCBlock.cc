#include "EventFilter/EcalRawToDigi/interface/DCCEETCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"


DCCEETCCBlock::DCCEETCCBlock ( DCCDataUnpacker  * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack) : 
DCCTCCBlock(u,m,e,unpack)
{
   blockLength_ = 0;
  
}

void DCCEETCCBlock::updateCollectors(){
  tps_ = unpacker_->ecalTpsCollection();
  pss_ = unpacker_->ecalPSsCollection();
}

void DCCEETCCBlock::addTriggerPrimitivesToCollection(){
  
  //point to trigger data
  data_++;

  bool processTPG2(true) ;
  uint psInputCounter(0);
  
  uint16_t * tccP_= reinterpret_cast< uint16_t * >(data_);
  
  
  int dccFOV =  event_->fov();
  if (! (dccFOV==dcc_FOV_0 || dccFOV==dcc_FOV_1 || dccFOV==dcc_FOV_2) ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectEvent") 
	<<"\n FOV value in data is: " << dccFOV <<
	"At event: "  <<event_->l1A()<<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
	<<"\n TCC id "<<tccId_<<" FOV "<< dccFOV << " which is not a foreseen value. Setting it to: " << dcc_FOV_2;
    }
    dccFOV=dcc_FOV_2;
  }
  
  
  /////////////////////////
  // MC raw data based on CMS NOTE 2005/021 
  // (and raw data when FOV was unassigned, earlier than mid 2008)   
    if (dccFOV == dcc_FOV_0)
      {

	// Unpack TPG1 pseudostrip input block
	if(ps_){ 
	  for(uint i = 1; i<= NUMB_PSEUDOSTRIPS; i++, tccP_++, psInputCounter++){
	    pPS_= mapper_->getPSInputDigiPointer(tccId_,psInputCounter);
	    if(!pPS_) continue;
	    pPS_->setSampleValue(0, *tccP_ );
	    (*pss_)->push_back(*pPS_);
	  }
	}
  
	// Unpack TPG1 trigger primitive block
	// loop over tp_counter=i and navigate forward in raw (tccP_)
	for(uint i = 1; i<= NUMB_TTS_TPG1; i++, tccP_++){//16
	  
	  if( i<= nTTs_){
	    pTP_ = mapper_->getTPPointer(tccId_,i);  // pointer to tp digi container
	    if(pTP_) pTP_->setSample(0, *tccP_ );    // fill it
	  }else {
	    processTPG2 = false;
	    break;
	  }
	  // adding trigger primitive digi to the collection
	  (*tps_)->push_back(*pTP_);
	}
	
	
	if(processTPG2){
	  
	  // Unpack TPG2 pseudostrip input block
	  if(ps_){ 
	    for(uint i = 1; i<= NUMB_PSEUDOSTRIPS; i++, tccP_++, psInputCounter++){
	      if (i<=NUMB_TTS_TPG2_DUPL) continue;
	      //fill pseudostrip container
	      pPS_= mapper_->getPSInputDigiPointer(tccId_,psInputCounter);
	      if(!pPS_) continue;
	      pPS_->setSampleValue(0, *tccP_ );
	      (*pss_)->push_back(*pPS_);
	    }
	  }
	  
	  
	  // Unpack TPG2 trigger primitive block
	  for(uint i = 1; i<= NUMB_TTS_TPG2; i++, tccP_++){//12
	    uint tt = i+NUMB_TTS_TPG1;
	    
	    if( tt <= nTTs_){
	      pTP_ = mapper_->getTPPointer(tccId_,tt);
	      if(pTP_) pTP_->setSample(0, *tccP_ ); 
	    }
	    else break;
	    // adding trigger primitive digi to the collection
	    (*tps_)->push_back(*pTP_);
	  }
	  
	}// end if(processTPG2)
	
      }// end FOV==0
    
    ///////////////////////////
      // real data since ever FOV was initialized; only 2 used >= June 09  
    else if (dccFOV == dcc_FOV_1  || dccFOV == dcc_FOV_2) 
      {
	
	// Unpack TPG1 pseudostrip input block
	if(ps_){ 
	  for(uint i = 1; i<= NUMB_PSEUDOSTRIPS; i++, tccP_++, psInputCounter++){
	    pPS_= mapper_->getPSInputDigiPointer(tccId_,psInputCounter);
	    if(!pPS_) continue;
	    pPS_->setSampleValue(0, *tccP_ );
	    (*pss_)->push_back(*pPS_);
	  }
	}
	
	int offset(0);
	// Unpack TPG1 trigger primitive block
	// loop over tp_counter=i and navigate forward in raw (tccP_)
	for(uint i = 1; i<= NUMB_TTS_TPG1; i++, tccP_++){//16
	  
	  if( i<= nTTs_){
	    if(mapper_->isTCCExternal(tccId_)){
	      if(i>8 && i<=16 ) continue;   // skip blank tp's [9,16]
	      if(i>8  )         offset=8;   // 
	      if(i>24 )         continue;   // skip blank tp's [25, 28]
	    }
	    
	    pTP_ = mapper_->getTPPointer(tccId_,i-offset);  // pointer to tp digi container
	    if(pTP_) pTP_->setSample(0, *tccP_ );    // fill it
	    
	  }else {
	    processTPG2 = false;
	    break;
	  }
	  // adding trigger primitive digi to the collection
	  (*tps_)->push_back(*pTP_);
	}
	
	if(processTPG2){
	  
	  // Unpack TPG2 pseudostrip input block
	  if(ps_){ 
	    for(uint i = 1; i<= NUMB_PSEUDOSTRIPS; i++, tccP_++, psInputCounter++){
	      if (i<=NUMB_TTS_TPG2_DUPL) continue;
	      //fill pseudostrip container
	      pPS_= mapper_->getPSInputDigiPointer(tccId_,psInputCounter);
	      if(!pPS_) continue;
	      pPS_->setSampleValue(0, *tccP_ );
	      (*pss_)->push_back(*pPS_);
	    }
	  }
	  
	  
	  // Unpack TPG2 trigger primitive block
	  for(uint i = 1; i<= NUMB_TTS_TPG2; i++, tccP_++){//12
	    uint tt = i+NUMB_TTS_TPG1;
	    
	    if( i<= nTTs_){
	      if(mapper_->isTCCExternal(tccId_)){
		if(tt>8 && tt<=16 ) continue;   // skip blank tp's [9,16]
		if(tt>8  )          offset=8;    // 
		if(tt>24 )          continue;    // skip blank tp's [25, 28]
	      }
	      
	      pTP_ = mapper_->getTPPointer(tccId_,tt-offset);
	      if(pTP_) pTP_->setSample(0, *tccP_ ); 
	    }
	    else break;
	    // adding trigger primitive digi to the collection
	    (*tps_)->push_back(*pTP_);
	  }
	}// end if(processTPG2)
      }// end FOV==1 or 2
}


bool DCCEETCCBlock::checkTccIdAndNumbTTs(){
  
	
  bool tccFound(false);
  bool errorOnNumbOfTTs(false);
  const int  activeDCC =  mapper_->getActiveSM();
  std::vector<uint> * m = mapper_->getTccs(activeDCC);
  std::vector<uint>::iterator it;
  for(it= m->begin();it!=m->end();it++){
    if((*it) == tccId_){
      tccFound=true;
     
	  /*
	    expNumbTTs_= 28; //separate from inner and outer tcc 
	    For inner TCCs you expect 28 TTs (4 in phi, 7 in eta).
	    For outer TCCs you expect 16 TTs (4 in phi, 4 in eta).
	    to implement : map tccid-> number of tts

            Pascal P.: For outer board, only 16 TPs are meaningful but 28 are sent.
            - [1,8] are meaningful 
            - [9,16] are empty ie should be skipped by unpacker 
            - [17, 24] are meaningful 
            - [25, 28] are empty ie should be skipped by unpacker 
	  */

	  if( nTTs_ != 28 && nTTs_ !=16 ){
	    if( ! DCCDataUnpacker::silentMode_ ){
          edm::LogWarning("IncorrectBlock") 
		   <<"\n Error on event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
           <<"\n TCC id "<<tccId_<<" has "<<nTTs_<<" Trigger Towers (only 28 or 16 are the expected values in EE)"
           <<"\n => Skipping to next fed block...";
          errorOnNumbOfTTs = true; 
          //todo : add to error collection   
	    }
	  }
    }
  }
	
  if(!tccFound){

    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectBlock") 
        <<"\n Error on event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in fed <<"<<mapper_->getActiveDCC()
        <<"\n TCC id "<<tccId_<<" is not valid for this dcc "
        <<"\n => Skipping to next fed block...";
       //todo : add to error collection   
     }
  }
  

 return (tccFound || errorOnNumbOfTTs);

}


uint DCCEETCCBlock::getLength(){

  uint64_t * temp = data_;
  temp++;

  ps_       = ( *temp>>TCC_PS_B ) & B_MASK;   
   
  uint numbTps = (NUMB_TTS_TPG1 + NUMB_TTS_TPG2);
  if(ps_){ numbTps += 2*NUMB_PSEUDOSTRIPS;}
  
  uint length = numbTps/4 + 2; //header and trailer
  if(numbTps%4) length++ ;
  
  
  return length;

}


