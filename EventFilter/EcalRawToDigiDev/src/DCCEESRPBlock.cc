
#include "EventFilter/EcalRawToDigiDev/interface/DCCEESRPBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataBlockPrototype.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/ECALUnpackerException.h"

#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"



DCCEESRPBlock::DCCEESRPBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack ) : 
DCCSRPBlock(u,m,e,unpack)
{}


void DCCEESRPBlock::updateCollectors(){
  // Set SR flag digis
  eeSrFlagsDigis_ = unpacker_->eeSrFlagsCollection(); 
}


void DCCEESRPBlock::addSRFlagToCollection(){
  
  // Point to SR flags 
  data_++;
  uint16_t * my16Bitp_ = reinterpret_cast<uint16_t *> (data_);

  
  for( uint n=0; n<expNumbSrFlags_ ;n++,pSCDetId_++ ){
   

    pSCDetId_ = mapper_->getSCDetIdPointer(n+1);   
   
    if( n!=0 && n%4==0 ) my16Bitp_++;
 
    if(pSCDetId_){ // this check is not really needed...
     ushort srFlag =  ( *my16Bitp_ >> ( (n-(n/4)*4) * 3 ) )  &  SRP_SRFLAG_MASK ;
     srFlags_[n] = srFlag;
     if(unpackInternalData_){
       EESrFlag sr(*pSCDetId_,srFlag);
       (*eeSrFlagsDigis_)->push_back(sr);
     }
    }else{break;}
     
  }
   
}

void DCCEESRPBlock::checkSrpIdAndNumbSRFlags(){

  expNumbSrFlags_=36;//to be corrected
   //todo :  to be implemented...


} 




