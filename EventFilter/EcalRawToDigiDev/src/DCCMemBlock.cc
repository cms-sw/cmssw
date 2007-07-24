
#include "EventFilter/EcalRawToDigiDev/interface/DCCMemBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include <stdio.h>
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"



DCCMemBlock::DCCMemBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e) 
:DCCDataBlockPrototype(u,m,e)
{

  unfilteredTowerBlockLength_  = mapper_->getUnfilteredTowerBlockLength();
  expXtalTSamples_             = mapper_->numbXtalTSamples();

  numbDWInXtalBlock_           = (expXtalTSamples_-2)/4+1;
  xtalBlockSize_               = numbDWInXtalBlock_*8;
  kSamplesPerPn_               = expXtalTSamples_*5;  
  
  uint numbOfXtalBlocks        = (unfilteredTowerBlockLength_-1)/numbDWInXtalBlock_; 
  uint numbOfPnBlocks          = numbOfXtalBlocks/5; //change 5 by a variable
  uint vectorSize              = numbOfPnBlocks*10*expXtalTSamples_;

  //Build pnDiodevector
  for(uint i =0; i< vectorSize; i++){ pn_.push_back(-1);}

}

void DCCMemBlock::updateCollectors(){

  invalidMemChIds_             = unpacker_->invalidMemChIdsCollection();
  invalidMemBlockSizes_        = unpacker_->invalidMemBlockSizesCollection();
  invalidMemTtIds_             = unpacker_->invalidMemTtIdsCollection();
  invalidMemGains_             = unpacker_->invalidMemGainsCollection();
  pnDiodeDigis_                = unpacker_->pnDiodeDigisCollection();

}



int DCCMemBlock::unpack(uint64_t ** data, uint * dwToEnd, uint expectedTowerID){
  
  error_   = false;  
  datap_   = data;
  data_    = *data;
  dwToEnd_ = dwToEnd;

 
  if( (*dwToEnd_)<1){
    edm::LogWarning("EcalRawToDigi@SUB=DCCTCCBlock:unpack")
      <<"EcalRawToDigi@SUB=DCCMemBlock"
      <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
      <<"\nThe end of event was reached !";
    return STOP_EVENT_UNPACKING;
  }
  
  lastStripId_     = 0;
  lastXtalId_      = 0;
  expTowerID_      = expectedTowerID;
  
  
  //Point to begin of block
  data_++;
  
  towerId_           = ( *data_ ) & TOWER_ID_MASK;
  nTSamples_         = ( *data_>>TOWER_NSAMP_B  ) & TOWER_NSAMP_MASK; 
  bx_                = ( *data_>>TOWER_BX_B     ) & TOWER_BX_MASK;
  l1_                = ( *data_>>TOWER_L1_B     ) & TOWER_L1_MASK;
  blockLength_       = ( *data_>>TOWER_LENGTH_B ) & TOWER_LENGTH_MASK;
 
  
  
  //cout<<"\n Tower id "<<dec<<towerId_<<" , Data pointer to TOWER BLOCK "<<hex<<(*data_)<<dec<<endl;
 
  //debugging
  //display(cout);

  // Block Length Check (1)
  if ( unfilteredTowerBlockLength_ != blockLength_ ){    
   
    // chosing channel 1 as representative of a dummy...
    EcalElectronicsId id(mapper_->getActiveSM(), expTowerID_,1, 1);
    
	edm::LogWarning("EcalRawToDigi@SUB=DCCMemBlockEcalRawToDigi@SUB=DCCTCCBlock:unpack")
      <<"EcalRawToDigi@SUB=DCCMemBlock"
      <<"\nFor event "<<event_->l1A()<<", dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
   	  <<"\nExpected block size is "<<(unfilteredTowerBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found";
    (*invalidMemBlockSizes_)->push_back(id);
	return STOP_EVENT_UNPACKING;
	 
  }

  // Block Length Check (2)
  if((*dwToEnd_)<blockLength_){
    edm::LogWarning("EcalRawToDigi@SUB=DCCMemBlockEcalRawToDigi@SUB=DCCTCCBlock:unpack")
	  <<"EcalRawToDigi@SUB=DCCMemBlock::unpack"
      <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
	  <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<(blockLength_*8)<<" are needed!";
	//Note : add to error collection 
    //could this be the same collection as previous case, i.e. invalidMemBlockSizes_  ?
	return STOP_EVENT_UNPACKING;
  }
  
  // Synchronization Check 
  if(sync_){
    uint dccBx = ( event_->l1A())&TOWER_BX_MASK;
    uint dccL1 = ( event_->bx() )&TOWER_L1_MASK;
    if( dccBx != bx_ || dccL1 != l1_ ){
      edm::LogWarning("EcalRawToDigi@SUB=DCCMemBlock::unpack")
        <<"\nSynchronization error for Mem block in event "<<event_->l1A()<<" with bx "<<event_->bx()<<" in dcc <<"<<mapper_->getActiveDCC()
        <<"\nMem local l1A is  "<<l1_<<" Mem local bx is "<<bx_;
        //Note : add to error collection ?
        // need of a new collection
      return STOP_EVENT_UNPACKING;
    }
  }  
    
  // Number Of Samples Check
  if( nTSamples_ != expXtalTSamples_ ){
    edm::LogWarning("EcalRawToDigi@SUB=DCCMemBlock::unpack")
      <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in dcc <<"<<mapper_->getActiveDCC()
      <<"\nNumber of time samples "<<nTSamples_<<" is not the same as expected ("<<expXtalTSamples_<<")";
    //Note : add to error collection ?		 
    return STOP_EVENT_UNPACKING;
  }
  
  
  //Channel Id Check
  if( expTowerID_ != towerId_){
    
    // chosing channel 1 as representative as a dummy...
    EcalElectronicsId id( mapper_->getActiveSM(), expTowerID_, 1,1);
    
    edm::LogWarning("EcalRawToDigiDevTowerId")
    <<"\nFor event "<<event_->l1A()<<" and dcc "<<mapper_->getActiveDCC()
    <<"\nExpected trigger tower is "<<expTowerID_<<" while "<<towerId_<<" was found ";
    
    (*invalidMemTtIds_)->push_back(id);
    towerId_=expTowerID_;
    
    // todo : go to the next mem
    error_= true;
	
	updateEventPointers();
	return SKIP_BLOCK_UNPACKING;
  }
   
 
  //point to xtal data
  data_++;
  
  //LogInfo("EcalRawToDigi")
  // <<"\n Tower id "<<dec<<towerId_<<" , Data pointer to XTAL DATA "<<hex<<(*data_)<<dec<<endl;
		               
  
  unpackMemTowerData();
  
  if(!error_){ fillPnDiodeDigisCollection();}

  updateEventPointers();
  
  return BLOCK_UNPACKED;
     
}



void DCCMemBlock::unpackMemTowerData(){
  
    
  //todo: move EcalPnDiodeDetId to electronics mapper


  lastTowerBeforeMem_ = 0;
  // differentiating the barrel and the endcap case
  if (9 < mapper_->getActiveSM() || mapper_->getActiveSM() < 46){
    lastTowerBeforeMem_ = 69; }
  else {
    lastTowerBeforeMem_ = 69; } 
  

  for(uint expStripId = 1; expStripId<= 5; expStripId++){

    for(uint expXtalId = 1; expXtalId <= 5; expXtalId++){ 
	 
      uint16_t * xData_= reinterpret_cast<uint16_t *>(data_);
 
      // Get xtal data ids
      uint stripId = (*xData_) & TOWER_STRIPID_MASK;
      uint xtalId  =((*xData_)>>TOWER_XTALID_B ) & TOWER_XTALID_MASK;
   
      bool errorOnDecoding(false);
	  
      if(expStripId != stripId || expXtalId != xtalId){ 
        // chosing channel and strip as EcalElectronicsId
        EcalElectronicsId id(mapper_->getActiveSM(), towerId_, expStripId, expXtalId);
        edm::LogWarning("EcalRawToDigiDevChId")
          <<"\nFor event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\nThe expected strip is "<<expStripId<<" and "<<stripId<<" was found"
          <<"\nThe expected xtal  is "<<expXtalId <<" and "<<xtalId<<" was found";

        stripId = expStripId;
   	    xtalId  = expXtalId;
		 
       (*invalidMemChIds_)->push_back(id);
	    errorOnDecoding = true; 
	
	    //Note : move to the next ...   
		 
     }
	 
     uint ipn, index;
		
     if((stripId-1)%2==0){ ipn = (towerId_-lastTowerBeforeMem_)*5 + xtalId - 1; }
     else                { ipn = (towerId_-lastTowerBeforeMem_)*5 + 5 - xtalId; }
	 
	  	
      //Cooking samples
      for(uint i =0; i< nTSamples_ ;i++){ 
      
        xData_++;
		  
        index = ipn*50 + (stripId-1)*nTSamples_+i;
		 
	    //edm::LogDebug("EcalRawToDigiDevChId")<<"\n Strip id "<<std::dec<<stripId<<" Xtal id "<<xtalId
	    //  <<" tsamp = "<<i<<" 16b = 0x "<<std::hex<<(*xData_)<<dec;
	   
        uint temp = (*xData_)&TOWER_DIGI_MASK;
		
  	     short sample(0);
		
		
        if( (stripId-1)%2 ) {
	     
          // If strip number is even, 14 bits are reversed in order
	       for(int ib=0;ib<14;ib++){ 
	         sample <<= 1;
	         sample |= (temp&1);
	         temp  >>= 1;
	       }
			
        } else { sample=temp;}
	
	     sample   ^=  0x800;
        uint gain =  sample>>12;
		 
		 
	     /* 
	       LogInfo("EcalRawToDigi")
	       <<"\n After cooking... "<<" 12b = 0x "<<hex<<(sample&0xFFF)<<" "<<dec<<(sample&0xFFF)<<" gain is "<<gain<<endl;;
	     */
			
        if( gain >= 2 ){
		  
          EcalElectronicsId id(mapper_->getActiveDCC(), towerId_, stripId,xtalId);
	  
	      edm::LogWarning("EcalRawToDigiDevGainZero")
	       <<"\nFor event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" ,tower "<<towerId_
	       <<"\nIn strip "<<stripId<<" xtal "<<xtalId<<" the gain is "<<gain<<" in sample "<<(i+1);

			 
          (*invalidMemGains_)->push_back(id);
          errorOnDecoding=true;
        }
		
        if( !errorOnDecoding && !error_){pn_[index]=sample;} //Note : move to the next versus flag...
		 
      }// loop over samples ended
	
      data_ += numbDWInXtalBlock_;
    }//loop over xtals
  }// loop over strips
	 

}

void DCCMemBlock::fillPnDiodeDigisCollection(){
	

  //todo change pnId max
  for (int pnId=1; pnId<=5; pnId++){
    bool errorOnPn(false);
    uint realPnId = pnId;
    
    if(towerId_==70){ realPnId += 5;}
	 
    // Note : we are assuming always 5 VFE channels enabled 
    // This means we all have 5 pns per tower 


    // temporary solution before sending creation of PnDigi's in mapper as done with crystals
    //     mapper_->getActiveSM()  : this is the 'dccid', number ranging internally in ECAL from 1 to 54
    //     mapper_->getActiveDCC() : this is the FED_id
    int ism;
    int dccId = mapper_->getActiveSM();
    
    if        (9< dccId && dccId < 28){
      ism  = dccId-9+18;}
    
    else if (27 < dccId && dccId< 46){
      ism  = dccId-9-18;}
    
    else
      {ism = -999;}
    
    
    //    EcalPnDiodeDetId PnId(EcalBarrel, mapper_->getActiveSM(), realPnId );
    // using ism insead of DCCId, to locate pn in the same place as the crystals that receive same laser pulses
    EcalPnDiodeDetId PnId(EcalBarrel, ism , realPnId );
    


    EcalPnDiodeDigi thePnDigi(PnId );
    thePnDigi.setSize(kSamplesPerPn_);
	 
	
    for (uint ts =0; ts <kSamplesPerPn_; ts++){
      
      short pnDiodeData = pn_[(towerId_-lastTowerBeforeMem_)*250 + (pnId-1)*kSamplesPerPn_ + ts];
      if( pnDiodeData == -1){
        errorOnPn=true;
	     break;
      }
	 
      EcalFEMSample thePnSample(pnDiodeData );
      thePnDigi.setSample(ts, thePnSample );  
    }
    
    if(!errorOnPn){ (*pnDiodeDigis_)->push_back(thePnDigi);}
  
  }
  
} 



void DCCMemBlock::display(std::ostream& o){

  o<<"\n Unpacked Info for DCC MEM Block"
  <<"\n DW1 ============================="
  <<"\n Tower Id "<<towerId_
  <<"\n Numb Samp "<<nTSamples_
  <<"\n Bx "<<bx_
  <<"\n L1 "<<l1_
  <<"\n blockLength "<<blockLength_;  
} 




