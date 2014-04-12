#include "EventFilter/EcalRawToDigi/interface/DCCMemBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include <stdio.h>
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"



DCCMemBlock::DCCMemBlock( DCCDataUnpacker * u,EcalElectronicsMapper * m, DCCEventBlock * e) 
:DCCDataBlockPrototype(u,m,e)
{

  unfilteredTowerBlockLength_  = mapper_->getUnfilteredTowerBlockLength();
  expXtalTSamples_             = mapper_->numbXtalTSamples();

  numbDWInXtalBlock_           = (expXtalTSamples_-2)/4+1;
  xtalBlockSize_               = numbDWInXtalBlock_*8;
  kSamplesPerPn_               = expXtalTSamples_*5;  
  
  unsigned int numbOfXtalBlocks        = (unfilteredTowerBlockLength_-1)/numbDWInXtalBlock_; 
  unsigned int numbOfPnBlocks          = numbOfXtalBlocks/5; //change 5 by a variable
  unsigned int vectorSize              = numbOfPnBlocks*10*expXtalTSamples_;

  //Build pnDiodevector
  for(unsigned int i =0; i< vectorSize; i++){ pn_.push_back(-1);}

}

void DCCMemBlock::updateCollectors(){

  invalidMemChIds_             = unpacker_->invalidMemChIdsCollection();
  invalidMemBlockSizes_        = unpacker_->invalidMemBlockSizesCollection();
  invalidMemTtIds_             = unpacker_->invalidMemTtIdsCollection();
  invalidMemGains_             = unpacker_->invalidMemGainsCollection();
  pnDiodeDigis_                = unpacker_->pnDiodeDigisCollection();

}



int DCCMemBlock::unpack(const uint64_t ** data, unsigned int * dwToEnd, unsigned int expectedTowerID){
  
  error_   = false;  
  datap_   = data;
  data_    = *data;
  dwToEnd_ = dwToEnd;

 
  if( (*dwToEnd_)<1){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectEvent")
        <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\nThe end of event was reached !";
    }
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
 
  event_->setFESyncNumbers(l1_,bx_,short(expectedTowerID-1));

 
  //debugging
  //display(cout);

  // Block Length Check (1)
  if ( unfilteredTowerBlockLength_ != blockLength_ ){    
   
    // chosing channel 1 as representative of a dummy...
    EcalElectronicsId id( mapper_->getActiveSM() , expTowerID_,1, 1);
    (*invalidMemBlockSizes_)->push_back(id);
    if( ! DCCDataUnpacker::silentMode_ ){ 
      edm::LogWarning("IncorrectEvent")
        <<"\nFor event "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower block "<<towerId_
        <<"\nExpected mem block size is "<<(unfilteredTowerBlockLength_*8)<<" bytes while "<<(blockLength_*8)<<" was found";
    }
    return STOP_EVENT_UNPACKING;
    
  }
  
  // Block Length Check (2)
  if((*dwToEnd_)<blockLength_){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectEvent")
        <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\n Only "<<((*dwToEnd_)*8)<<" bytes are available while "<<(blockLength_*8)<<" are needed!";
      // chosing channel 1 as representative of a dummy...
    } 
    EcalElectronicsId id( mapper_->getActiveSM() , expTowerID_,1, 1);
    (*invalidMemBlockSizes_)->push_back(id);
    return STOP_EVENT_UNPACKING;
  }
  
  // Synchronization Check 
  if(sync_){
    const unsigned int dccBx = ( event_->bx()) & TOWER_BX_MASK;
    const unsigned int dccL1 = ( event_->l1A() ) & TOWER_L1_MASK;
    const unsigned int fov   = ( event_->fov() ) & H_FOV_MASK;
    
    if (! isSynced(dccBx, bx_, dccL1, l1_, FE_MEM, fov)) {
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("IncorrectEvent")
          << "Synchronization error for Mem block"
          << " (L1A " << event_->l1A() << " bx " << event_->bx() << " fed " << mapper_->getActiveDCC() << ")\n"
          << "  dccBx = " << dccBx << " bx_ = " << bx_ << " dccL1 = " << dccL1 << " l1_ = " << l1_ << "\n"
          << "  => Stop event unpacking";
      }
      //Note : add to error collection ?
      // need of a new collection
      return STOP_EVENT_UNPACKING;
    }
  }  
  
  // Number Of Samples Check
  if( nTSamples_ != expXtalTSamples_ ){
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectEvent")
        <<"\nUnable to unpack MEM block for event "<<event_->l1A()<<" in fed "<<mapper_->getActiveDCC()
        <<"\nNumber of time samples "<<nTSamples_<<" is not the same as expected ("<<expXtalTSamples_<<")";
     }
    //Note : add to error collection ?		 
    return STOP_EVENT_UNPACKING;
  }
  
  
  //Channel Id Check
  if( expTowerID_ != towerId_){
    
    // chosing channel 1 as representative as a dummy...
    EcalElectronicsId id( mapper_->getActiveSM() , expTowerID_, 1,1);
    (*invalidMemTtIds_)->push_back(id);
    if( ! DCCDataUnpacker::silentMode_ ){
      edm::LogWarning("IncorrectBlock")
        <<"For event "<<event_->l1A()<<" and fed "<<mapper_->getActiveDCC() << " and sm: "  << mapper_->getActiveSM()
        <<"\nExpected mem tower block is "<<expTowerID_<<" while "<<towerId_<<" was found ";
     }
    
    towerId_=expTowerID_;
    
    // todo : go to the next mem
    error_= true;
	
	updateEventPointers();
	return SKIP_BLOCK_UNPACKING;
  }
   
 
  //point to xtal data
  data_++;
		               
  
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
  

  for(unsigned int expStripId = 1; expStripId<= 5; expStripId++){

    for(unsigned int expXtalId = 1; expXtalId <= 5; expXtalId++){
	 
      const uint16_t * xData_= reinterpret_cast<const uint16_t *>(data_);
 
      // Get xtal data ids
      unsigned int stripId = (*xData_) & TOWER_STRIPID_MASK;
      unsigned int xtalId  =((*xData_)>>TOWER_XTALID_B ) & TOWER_XTALID_MASK;
   
      bool errorOnDecoding(false);
	  
      if(expStripId != stripId || expXtalId != xtalId){

        // chosing channel and strip as EcalElectronicsId
        EcalElectronicsId id( mapper_->getActiveSM() , towerId_, expStripId, expXtalId);
       (*invalidMemChIds_)->push_back(id);
      
        if( ! DCCDataUnpacker::silentMode_ ){
          edm::LogWarning("IncorrectBlock")
            <<"For event "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower mem block "<<towerId_
            <<"\nThe expected strip is "<<expStripId<<" and "<<stripId<<" was found"
            <<"\nThe expected xtal  is "<<expXtalId <<" and "<<xtalId<<" was found";
        }

        stripId = expStripId;
        xtalId  = expXtalId;
		 


         errorOnDecoding = true; 
	
       //Note : move to the next ...   
		 
     }
	 
     unsigned int ipn, index;
		
     if((stripId-1)%2==0){ ipn = (towerId_-lastTowerBeforeMem_)*5 + xtalId - 1; }
     else                { ipn = (towerId_-lastTowerBeforeMem_)*5 + 5 - xtalId; }
	 
	  	
      //Cooking samples
      for(unsigned int i =0; i< nTSamples_ ;i++){ 
      
        xData_++;
		  
        index = ipn*50 + (stripId-1)*nTSamples_+i;
		 
	    //edm::LogDebug("EcalRawToDigiMemChId")<<"\n Strip id "<<std::dec<<stripId<<" Xtal id "<<xtalId
	    //  <<" tsamp = "<<i<<" 16b = 0x "<<std::hex<<(*xData_)<<dec;
	   
        unsigned int temp = (*xData_)&TOWER_DIGI_MASK;
		
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
        unsigned int gain =  sample>>12;
			
        if( gain >= 2 ){

          EcalElectronicsId id(mapper_->getActiveSM() , towerId_, stripId,xtalId);
          (*invalidMemGains_)->push_back(id);
          
           if( ! DCCDataUnpacker::silentMode_ ){
	      edm::LogWarning("IncorrectGain")
	       <<"For event "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" , mem tower block "<<towerId_
	       <<"\nIn strip "<<stripId<<" xtal "<<xtalId<<" the gain is "<<gain<<" in sample "<<(i+1);
           }

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
    unsigned int realPnId = pnId;
    
    if(towerId_==70){ realPnId += 5;}
	 
    // Note : we are assuming always 5 VFE channels enabled 
    // This means we all have 5 pns per tower 

    // solution before sending creation of PnDigi's in mapper as done with crystals
    //     mapper_->getActiveSM()  : this is the 'dccid'
    //     number ranging internally in ECAL from 1 to 54, according convention specified here:
    //     http://indico.cern.ch/getFile.py/access?contribId=0&resId=0&materialId=slides&confId=11621

    //     mapper_->getActiveDCC() : this is the FED_id (601 - 654 for ECAL at CMS)

    const int activeSM = mapper_->getActiveSM();
    int subdet(0);
    if (NUMB_SM_EB_MIN_MIN <= activeSM && activeSM <= NUMB_SM_EB_PLU_MAX) {
      subdet = EcalBarrel;
    }
    else if( (NUMB_SM_EE_MIN_MIN <= activeSM && activeSM <= NUMB_SM_EE_MIN_MAX) ||
            (NUMB_SM_EE_PLU_MIN <= activeSM && activeSM <= NUMB_SM_EE_PLU_MAX) ) {
      subdet = EcalEndcap;
    }
    else {
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("IncorrectMapping")
          <<"\n mapper points to non existing dccid: " << activeSM;
      }
    }


    EcalPnDiodeDetId PnId(subdet, activeSM, realPnId );
    
    EcalPnDiodeDigi thePnDigi(PnId );
    thePnDigi.setSize(kSamplesPerPn_);
    
    
    for (unsigned int ts =0; ts <kSamplesPerPn_; ts++){
      
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
  <<"\n Mem Tower Block Id "<<towerId_
  <<"\n Numb Samp "<<nTSamples_
  <<"\n Bx "<<bx_
  <<"\n L1 "<<l1_
  <<"\n blockLength "<<blockLength_;  
} 




