#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalDCCHeaderRuntypeDecoder.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCFEBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCMemBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCTCCBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCSRPBlock.h"
#include <sys/time.h>

#include <iomanip>
#include <sstream>

DCCEventBlock::DCCEventBlock( DCCDataUnpacker * u , EcalElectronicsMapper * m ,  bool hU, bool srpU, bool tccU, bool feU, bool memU, bool forceToKeepFRdata) : 
  unpacker_(u), mapper_(m), headerUnpacking_(hU), srpUnpacking_(srpU), tccUnpacking_(tccU), feUnpacking_(feU),memUnpacking_(memU), forceToKeepFRdata_(forceToKeepFRdata)
{
  
  // Build a Mem Unpacker Block
  memBlock_   = new DCCMemBlock(u,m,this);
 
  // setup and initialize ch status vectors
  for( int feChannel=1;  feChannel <= 70;  feChannel++) { feChStatus_.push_back(0); hlt_.push_back(1);}
  for( int tccChannel=1; tccChannel <= 4 ; tccChannel++){ tccChStatus_.push_back(0);}
  
  // setup and initialize sync vectors
  for( int feChannel=1;  feChannel <= 70;  feChannel++) { feBx_.push_back(-1);  feLv1_.push_back(-1); }
  for( int tccChannel=1; tccChannel <= 4 ; tccChannel++){ tccBx_.push_back(-1); tccLv1_.push_back(-1);}
  srpBx_=-1;
  srpLv1_=-1;
  
}


void DCCEventBlock::reset(){

  // reset sync vectors
  for( int feChannel=1;  feChannel <= 70;  feChannel++) {   feBx_[feChannel-1]=-1;   feLv1_[feChannel-1]=-1; }
  for( int tccChannel=1; tccChannel <= 4 ; tccChannel++){ tccBx_[tccChannel-1]=-1; tccLv1_[tccChannel-1]=-1;}
  srpBx_=-1;
  srpLv1_=-1;


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



unsigned int DCCEventBlock::next_tower_search(const unsigned int current_tower_id)
{
  const uint64_t* const prev_data = data_;
  const unsigned int prev_dwToEnd = dwToEnd_;
  
  // expected LV1, BX, #TS
  const uint32_t lv1 = ((l1_ - 1) & 0xFFF);
  const uint32_t bx = (bx_ != 3564) ? bx_ : 0;
  const uint32_t ts = mapper_->numbXtalTSamples();
  
  // construct tower header and mask
  const uint64_t s_hi = 0xC0000000 + lv1;
  const uint64_t s_lo = 0xC0000000 + (bx << 16) + (ts << 8);
  
  const uint64_t sign = (s_hi << 32) + s_lo;
  const uint64_t mask = 0xC0001FFFDFFF7F00;
  
  // step forward to skip header word of problematic tower
  data_++;
  dwToEnd_--;
  
  //std::cerr << "header of bad tower = " << current_tower_id << " #" << dwToEnd_ << " 0x" << std::hex << *data_ << std::dec << std::endl;
  //std::cerr << "mask and sign = 0x" << std::hex << mask << " 0x" << sign << std::dec << std::endl;
  
  // navigate through tower data blocks to find tower block header
  while (dwToEnd_ > 0) {
    data_++;
    dwToEnd_--;
    
    //std::cerr << current_tower_id << " #" << dwToEnd_ << " 0x" << std::hex << *data_ << " 0x" << (*data_ & mask) << std::dec << std::endl;
    
    if ((*data_ & mask) == sign) {
      const unsigned int next_tower_id = (*data_) & 0xFF;
      
      if (next_tower_id <= current_tower_id) continue;
      
      //std::cerr << "next tower = " << next_tower_id << std::endl;
      
      // step back one word of the next tower header
      data_--;
      dwToEnd_++;
      
      return next_tower_id;
    }
  }
  
  // can't find next tower header
  // restore data pointer
  data_ = prev_data;
  dwToEnd_ = prev_dwToEnd;
  return 1000;
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
  
  
  theDCCheader.setSRPLv1(srpLv1_);
  theDCCheader.setSRPBx(srpBx_);
  theDCCheader.setFELv1(feLv1_);
  theDCCheader.setFEBx(feBx_);
  theDCCheader.setTCCLv1(tccLv1_);
  theDCCheader.setTCCBx(tccBx_);
  

  EcalDCCHeaderRuntypeDecoder theRuntypeDecoder;
  unsigned int DCCruntype              = runType_;
  unsigned int DCCdetTriggerType = detailedTriggerType_;
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


// -----------------------------------------------------------------------
// sync checking

bool isSynced(const unsigned int dccBx,
              const unsigned int bx,
              const unsigned int dccL1,
              const unsigned int l1,
              const BlockType type,
              const unsigned int fov)
{
  // avoid checking for MC until EcalDigiToRaw bugfixed
  // and to guarantee backward compatibility on RAW data
  if ( fov < 1 ) return true;
  // check the BX sync according the following rule:
  //
  //   FE Block     MEM Block     TCC Block  SRP Block  DCC
  // ------------------------------------------------------------------
  //   fe_bx     == mem_bx == 0   tcc_bx ==  srp_bx ==  DCC_bx == 3564
  //   fe_bx     == mem_bx     == tcc_bx ==  srp_bx ==  DCC_bx != 3564
  
  const bool bxSynced =
    ((type ==  FE_MEM) && (bx ==     0) && (dccBx == 3564)) ||
    ((type ==  FE_MEM) && (bx == dccBx) && (dccBx != 3564)) ||
    ((type == TCC_SRP) && (bx == dccBx));
  
  // check the L1A sync:
  //
  // L1A counter relation is valid modulo 0xFFF:
  // fe_l1  == mem_l1 == (DCC_l1-1) & 0xFFF
  // tcc_l1 == srp_l1 ==  DCC_l1    & 0xFFF
  
  const bool l1Synced =
    ((type ==  FE_MEM) && (l1 == ((dccL1 - 1) & 0xFFF))) ||
    ((type == TCC_SRP) && (l1 == ( dccL1      & 0xFFF)));
  
  return (bxSynced && l1Synced);
}
