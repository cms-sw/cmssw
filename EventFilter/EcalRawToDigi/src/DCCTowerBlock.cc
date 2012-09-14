#include <stdio.h>
#include <algorithm>
#include "EventFilter/EcalRawToDigi/interface/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"



DCCTowerBlock::DCCTowerBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack , bool forceToKeepFRdata)
: DCCFEBlock(u,m,e,unpack,forceToKeepFRdata){}


void DCCTowerBlock::updateCollectors(){

  DCCFEBlock::updateCollectors();
  
  // needs to be update for eb/ee
  digis_                 = unpacker_->ebDigisCollection();
   
  invalidGains_          = unpacker_->invalidGainsCollection();
  invalidGainsSwitch_    = unpacker_->invalidGainsSwitchCollection();
  invalidChIds_          = unpacker_->invalidChIdsCollection();

}



int DCCTowerBlock::unpackXtalData(unsigned int expStripID, unsigned int expXtalID){
  
  bool errorOnXtal(false);
 
  const uint16_t * xData_= reinterpret_cast<const uint16_t *>(data_);

  // Get xtal data ids
  unsigned int stripId  = (*xData_)                     & TOWER_STRIPID_MASK;
  unsigned int xtalId   = ((*xData_)>>TOWER_XTALID_B )  & TOWER_XTALID_MASK;

  // check id in case data are not 0suppressed
  if( !zs_ && (expStripID != stripId || expXtalID != xtalId)){ 
    if(! DCCDataUnpacker::silentMode_){ 
      edm::LogWarning("IncorrectBlock")
        <<"For event L1A: "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
        <<"\n The expected strip is "<<expStripID<<" and "<<stripId<<" was found"
        <<"\n The expected xtal  is "<<expXtalID <<" and "<<xtalId<<" was found";        
    } 
    // using expected cry_di to raise warning about xtal_id problem
    pDetId_ = (EBDetId*) mapper_->getDetIdPointer(towerId_,expStripID,expXtalID);
    (*invalidChIds_)->push_back(*pDetId_);
    
    stripId    = expStripID;
    xtalId     = expXtalID;
    errorOnXtal= true;
    
    // return here, so to skip all following checks
    lastXtalId_++;
    if (lastXtalId_ > NUMB_XTAL)         {lastXtalId_=1; lastStripId_++;}
    data_ += numbDWInXtalBlock_;
    return BLOCK_UNPACKED;
  }


  // check id in case of 0suppressed data

  else if(zs_){

    // Check for valid Ids 1) values out of range

    if(stripId == 0 || stripId > 5 || xtalId == 0 || xtalId > 5){
      if( ! DCCDataUnpacker::silentMode_ ){
        edm::LogWarning("IncorrectBlock")
          <<"For event L1A: "<<event_->l1A()<<", fed "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n Invalid strip : "<<stripId<<" or xtal : "<<xtalId
          <<" ids ( last strip was: " << lastStripId_ << " last ch was: " << lastXtalId_ << ")";
      }
      
      int st = lastStripId_;
      int ch = lastXtalId_;
      ch++;
      if (ch > NUMB_XTAL)         {ch=1; st++;}
      if (st > NUMB_STRIP)        {ch=1; st=1;}
      
      // adding channel following the last valid
      //pDetId_ = (EBDetId*) mapper_->getDetIdPointer(towerId_,st,ch);
      //(*invalidChIds_)->push_back(*pDetId_);
      fillEcalElectronicsError(invalidZSXtalIds_); 

      errorOnXtal = true;

      lastStripId_ = st;
      lastXtalId_  = ch;
      
      // return here, so to skip all following checks
      return SKIP_BLOCK_UNPACKING;
      
    }else{

      // Check for zs valid Ids 2) if channel-in-strip has increased wrt previous xtal
      
      
      // Check for zs valid Ids 2) if channel-in-strip has increased wrt previous xtal
      //                        3) if strip has increased wrt previous xtal
      if( ( stripId == lastStripId_ && xtalId <= lastXtalId_ ) ||
          (stripId < lastStripId_))
        {
          if (! DCCDataUnpacker::silentMode_) {
            edm::LogWarning("IncorrectBlock")
              << "Xtal id was expected to increase but it didn't - last valid xtal id was " << lastXtalId_ << " while current xtal is " << xtalId
              << " (LV1 " << event_->l1A() << " fed " << mapper_->getActiveDCC() << " tower " << towerId_ << ")";
          }
          
          int st = lastStripId_;
          int ch = lastXtalId_;
          ch++;
          if (ch > NUMB_XTAL)        {ch=1; st++;}
          if (st >  NUMB_STRIP)        {ch=1; st=1;}
          
          // adding channel following the last valid
          //pDetId_ = (EBDetId*) mapper_->getDetIdPointer(towerId_,st,ch);
          //(*invalidChIds_)->push_back(*pDetId_);
          fillEcalElectronicsError(invalidZSXtalIds_); 
          
          errorOnXtal = true;
          lastStripId_ = st;
          lastXtalId_  = ch;

          // return here, so to skip all following checks
          return SKIP_BLOCK_UNPACKING;
          
        }
      
      // if channel id not proven wrong, update lastStripId_ and lastXtalId_
      lastStripId_  = stripId;
      lastXtalId_   = xtalId;
    }//end else
  }// end if (zs_)


  bool addedFrame=false;

  // if there is an error on xtal id ignore next error checks  
  // otherwise, assume channel_id is valid and proceed with making and checking the data frame
  if(errorOnXtal) return SKIP_BLOCK_UNPACKING;

  pDetId_ = (EBDetId*) mapper_->getDetIdPointer(towerId_,stripId,xtalId);
  (*digis_)->push_back(*pDetId_);
  EBDataFrame df( (*digis_)->back() );
  addedFrame=true;
  bool wrongGain(false);

  //set samples in the data frame
  for(unsigned int i =0; i< nTSamples_ ;i++){ // loop on samples 
    xData_++;
    unsigned int data =  (*xData_) & TOWER_DIGI_MASK;
    unsigned int gain =  data>>12;
    xtalGains_[i]=gain;
    if(gain == 0){ 
      wrongGain = true; 
      // although gain==0 found, produce the dataFrame in order to have it, for saturation case
    } 
    df.setSample(i,data);
  }// loop on samples        

  bool isSaturation(true);
  if(wrongGain){
    
    // check whether the gain==0 has features of saturation or not
    // gain==0 occurs either in case of data corruption or of ADC saturation
    //                                  \->reject digi            \-> keep digi
    
    // determine where gainId==0 starts
    short firstGainZeroSampID(-1);    short firstGainZeroSampADC(-1);
    for (unsigned int s=0; s<nTSamples_; s++ ) {
      if(df.sample(s).gainId()==0 && firstGainZeroSampID==-1)
        {
          firstGainZeroSampID  = s;
          firstGainZeroSampADC = df.sample(s).adc();
          break;
        }
    }
    
    // check whether gain==0 and adc() stays constant for (at least) 5 consecutive samples
    unsigned int plateauEnd = std::min(nTSamples_,(unsigned int)(firstGainZeroSampID+5));
    for (unsigned int s=firstGainZeroSampID; s<plateauEnd; s++) 
      {
        if( df.sample(s).gainId()==0 && df.sample(s).adc()==firstGainZeroSampADC ) {;}
        else
          { isSaturation=false;  break;}  //it's not saturation
      }
    // get rid of channels which are stuck in gain0
    if(firstGainZeroSampID<3) {isSaturation=false; }
    
    if (! DCCDataUnpacker::silentMode_) {
      if (unpacker_->getChannelValue(mapper_->getActiveDCC(), towerId_, stripId, xtalId) != 10) {
        edm::LogWarning("IncorrectGain")
          << "Gain zero" << (isSaturation ? " with features of saturation" : "" ) << " was found in Tower Block"
          << " (L1A " << event_->l1A() << " bx " << event_->bx() << " fed " << mapper_->getActiveDCC()
          << " tower " << towerId_ << " strip " << stripId << " xtal " << xtalId << ")";
      }
    }
    
    if (! isSaturation)
      {
        (*invalidGains_)->push_back(*pDetId_);
        (*digis_)->pop_back(); 
        errorOnXtal = true; 
        
        //Point to begin of next xtal Block
        data_ += numbDWInXtalBlock_;
        //return here, so to skip all the rest
        //make special collection for gain0 data frames when due to saturation
        return BLOCK_UNPACKED;
      }//end isSaturation 
    else {
        data_ += numbDWInXtalBlock_;
        return BLOCK_UNPACKED;
    }

    }//end WrongGain
  
  
  // from here on, care about gain switches
  
  short firstGainWrong=-1;
  short numGainWrong=0;
  
  for (unsigned int i=1; i<nTSamples_; i++ ) {
    if (i>0 && xtalGains_[i-1]>xtalGains_[i]) {
          numGainWrong++;
          if (firstGainWrong == -1) { firstGainWrong=i;}
    }
  }
  
  
  if (numGainWrong > 0) {
    if (! DCCDataUnpacker::silentMode_) {
      edm::LogWarning("IncorrectGain")
        << "A wrong gain transition switch was found for Tower Block in strip " << stripId << " and xtal " << xtalId
        << " (L1A " << event_->l1A() << " bx " << event_->bx() << " fed " << mapper_->getActiveDCC() << " tower " << towerId_ << ")";
    }
    
    (*invalidGainsSwitch_)->push_back(*pDetId_);
    errorOnXtal = true;
  }
  
  //Add frame to collection only if all data format and gain rules are respected
  if (errorOnXtal && addedFrame) {
    (*digis_)->pop_back();
  }
  
  //Point to begin of next xtal Block
  data_ += numbDWInXtalBlock_;
  
  return BLOCK_UNPACKED;
}



void DCCTowerBlock::fillEcalElectronicsError( std::auto_ptr<EcalElectronicsIdCollection> * errorColection ){

   const int activeDCC = mapper_->getActiveSM();

   if(NUMB_SM_EB_MIN_MIN<=activeDCC && activeDCC<=NUMB_SM_EB_PLU_MAX){
     EcalElectronicsId  *  eleTp = mapper_->getTTEleIdPointer(activeDCC+TCCID_SMID_SHIFT_EB,expTowerID_);
     (*errorColection)->push_back(*eleTp);
   }else{
      if( ! DCCDataUnpacker::silentMode_ ){
          edm::LogWarning("IncorrectBlock")
            <<"For event "<<event_->l1A()<<" there's fed: "<< activeDCC
            <<" activeDcc: "<<mapper_->getActiveSM()
            <<" but that activeDcc is not valid in EB.";
        }

   }

}

