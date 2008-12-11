#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEBEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEEEventBlock.h"


bool DCCDataUnpacker::silentMode_ = false;

DCCDataUnpacker::DCCDataUnpacker( 
  EcalElectronicsMapper * mapper, bool hU, bool srpU, bool tccU, bool feU , bool memU, bool syncCheck, bool feIdCheck
){ 
  electronicsMapper_ = mapper;
  ebEventBlock_   = new DCCEBEventBlock(this,mapper,hU,srpU,tccU,feU,memU);
  eeEventBlock_   = new DCCEEEventBlock(this,mapper,hU,srpU,tccU,feU,memU);
  if(syncCheck){
    ebEventBlock_->enableSyncChecks();  
    eeEventBlock_->enableSyncChecks();
  }
  if(feIdCheck){
    ebEventBlock_->enableFeIdChecks();  
    eeEventBlock_->enableFeIdChecks();
  }
}


void DCCDataUnpacker::unpack(uint64_t * buffer, uint bufferSize, uint smId, uint fedId){
  //buffer is pointer to binary data
  //See if this fed is on EB or in EE

  if(smId>9&&smId<46){ 
    
    currentEvent_      = ebEventBlock_;
    ebEventBlock_    ->updateCollectors();
    ebEventBlock_    ->unpack(buffer,bufferSize,fedId); 
	 
  }
  else{               
   
    currentEvent_     = eeEventBlock_;
    eeEventBlock_    ->updateCollectors();
    eeEventBlock_    ->unpack(buffer,bufferSize,fedId); 

  }
    
}

DCCDataUnpacker::~DCCDataUnpacker(){
  delete ebEventBlock_;
  delete eeEventBlock_;
}
