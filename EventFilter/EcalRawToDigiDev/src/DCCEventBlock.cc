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



void DCCEventBlock::enableFeIdChecks(){
   towerBlock_   ->enableFeIdChecks();
}



void DCCEventBlock::updateCollectors(){

  dccHeaders_  = unpacker_->dccHeadersCollection();

  memBlock_    ->updateCollectors(); 
  tccBlock_    ->updateCollectors();
  srpBlock_    ->updateCollectors();
  towerBlock_  ->updateCollectors();
  
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

