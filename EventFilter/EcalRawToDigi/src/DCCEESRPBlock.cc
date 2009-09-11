#include "EventFilter/EcalRawToDigi/interface/DCCEESRPBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataBlockPrototype.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"



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
   
    if( n!=0 && n%4==0 ) my16Bitp_++;
 
     ushort srFlag =  ( *my16Bitp_ >> ( (n-(n/4)*4) * 3 ) )  &  SRP_SRFLAG_MASK ;
     srFlags_[n] = srFlag;
     if(unpackInternalData_){
       std::vector<EcalSrFlag*> srs = mapper_->getSrFlagPointer(n+1);
       for(size_t i = 0; i < srs.size(); ++i){
         srs[i]->setValue(srFlag); 
         (*eeSrFlagsDigis_)->push_back(*((EESrFlag*)srs[i]));
       } 
     }  
  }
}

bool DCCEESRPBlock::checkSrpIdAndNumbSRFlags(){

  expNumbSrFlags_=36;//to be corrected

  int dccId = mapper_->getActiveDCC() - 600;
  if (dccId == SECTOR_EEM_CCU_JUMP || dccId == SECTOR_EEP_CCU_JUMP) expNumbSrFlags_ = 41;

  //todo :  checks to be implemented...
  return true;

} 

