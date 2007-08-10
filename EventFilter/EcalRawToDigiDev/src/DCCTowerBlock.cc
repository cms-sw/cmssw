
#include "EventFilter/EcalRawToDigiDev/interface/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/ECALUnpackerException.h"
#include <stdio.h>
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"



DCCTowerBlock::DCCTowerBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, DCCEventBlock * e, bool unpack )
: DCCFEBlock(u,m,e,unpack){}


void DCCTowerBlock::updateCollectors(){

  DCCFEBlock::updateCollectors();
  
  // needs to be update for eb/ee
  digis_                  = unpacker_->ebDigisCollection();
   
  invalidGains_           = unpacker_->invalidGainsCollection();
  invalidGainsSwitch_     = unpacker_->invalidGainsSwitchCollection();
  invalidGainsSwitchStay_ = unpacker_->invalidGainsSwitchStayCollection();
  invalidChIds_           = unpacker_->invalidChIdsCollection();
 

}



void DCCTowerBlock::unpackXtalData(uint expStripID, uint expXtalID){
  
  bool errorOnXtal(false);
 
  uint16_t * xData_= reinterpret_cast<uint16_t *>(data_);

  // Get xtal data ids
  uint stripId = (*xData_) & TOWER_STRIPID_MASK;
  uint xtalId  =((*xData_)>>TOWER_XTALID_B ) & TOWER_XTALID_MASK;
  
  // cout<<"\n DEBUG : unpacked xtal data for strip id "<<stripId<<" and xtal id "<<xtalId<<endl;
  // cout<<"\n DEBUG : expected strip id "<<expStripID<<" expected xtal id "<<expXtalID<<endl;
  
  pDetId_ = (EBDetId*) mapper_->getDetIdPointer(towerId_,stripId,xtalId);
  if (!pDetId_) {
   edm::LogWarning("EcalRawToDigiDevChId")
      <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
      <<"\n no Det Id for strip  "<< stripId <<" and xtal "<<xtalId<<" was found";
  }

  if( !zs_ && (expStripID != stripId || expXtalID != xtalId)){ 
    
    edm::LogWarning("EcalRawToDigiDevChId")
      <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
      <<"\n The expected strip is "<<expStripID<<" and "<<stripId<<" was found"
      <<"\n The expected xtal  is "<<expXtalID <<" and "<<xtalId<<" was found";	

    (*invalidChIds_)->push_back(*pDetId_);

    stripId = expStripID;
    xtalId  = expXtalID;
    errorOnXtal = true;
    // one could return here, so to skip all the rest
  }

  else if(zs_){

    // Check for valid Ids	 
    if(stripId == 0 || stripId > 5 || xtalId == 0 || xtalId > 5){
      edm::LogWarning("EcalRawToDigiDevChId")
        <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
        <<"\n Invalid strip : "<<stripId<<" or xtal : "<<xtalId<<" ids";	
      
      //Todo : add to error collection
      // one cannot really  say which channel has the problem...
      // in previous version all the 25 channels of the tower were assigned an integrity error
      
      errorOnXtal = true;
      // one could return here, so to skip all the rest
      
    }else{
	 
	 
      // check if strip and xtal id increases
      if ( stripId >= lastStripId_ ){
        if( stripId == lastStripId_ && xtalId < lastXtalId_ ){
                         	  // shouldn't  "<"  rather be  "=<" ?

          edm::LogWarning("EcalRawToDigiDevChId")
            <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
            <<"\n Xtal id was expected to increase but it didn't "
            <<"\n Last unpacked xtal was "<<lastXtalId_<<" while current xtal is "<<xtalId;

	  // one cannot really  say which channel has the problem...
	  // in previous version all the 25 channels of the tower were assigned an integrity error
	  (*invalidChIds_)->push_back(*pDetId_);
	  
	  errorOnXtal = true;
	  // one could return here, so to skip all the rest
	}
      }

      else if( stripId < lastStripId_){
      
        edm::LogWarning("EcalRawToDigiDevChId")
          <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n Strip id was expected to increase but it didn't "
          <<"\n Last unpacked strip was "<<lastStripId_<<" while current strip is "<<stripId;
 

	// one cannot really  say which channel has the problem...
	// in previous version all the 25 channels of the tower were assigned an integrity error
	(*invalidChIds_)->push_back(*pDetId_);
		
       errorOnXtal = true;
       // one could return here, so to skip all the rest
      }
		
      lastStripId_  = stripId;
      lastXtalId_   = xtalId;
    }
  }
 
  bool frameAdded=false;
  // if there is an error on xtal id ignore next error checks  
  // otherwise, assume channel_id is valid and proceed with making and checking the data frame
  if(!errorOnXtal){ 

     //     if(pDFId_){ //thisis not needed for the EB
     (*digis_)->push_back(*pDetId_);
     EBDataFrame df( (*digis_)->back() );
     frameAdded=true;
     bool wrongGain(false);
	 
     if ( nTSamples_!=df.size()) {
       edm::LogWarning("EcalRawToDigiDevWrongSize")
       <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
       <<"\n dataframe size " << df.size() << " inconsistent with #sample " << nTSamples_;
     }
      //set samples in the frame
     for(uint i =0; i< nTSamples_ ;i++){ 
       xData_++;
       uint data =  (*xData_) & TOWER_DIGI_MASK;
       uint gain =  data>>12;
       //xtalGain.push_back(gain);
       xtalGains_[i]=gain;
       if(gain == 0){ 
	 wrongGain = true; 
	 // one could continue here to skip part of the loop
	 // as well as add the error to the collection and write the message 
       } 
 
       df.setSample(i,data);
      }
	
    
      if(wrongGain){ 
        edm::LogWarning("EcalRawToDigiDevGainZero")
        <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
        <<"\n Gain zero was found in strip "<<stripId<<" and xtal "<<xtalId;   

        (*invalidGains_)->push_back(*pDetId_);
        errorOnXtal = true;
      }
	
   
      short firstGainWrong=-1;
      short numGainWrong=0;
	    
      for (uint i=0; i<nTSamples_; i++ ) {
        if (i>0 && xtalGains_[i-1]>xtalGains_[i]) {
          numGainWrong++;
          if (firstGainWrong == -1) { firstGainWrong=i;}
        }
      }
   
      bool wrongGainStaysTheSame=false;
   
      if (firstGainWrong!=-1 && firstGainWrong<9){
        short gainWrong = xtalGains_[firstGainWrong];
        // does wrong gain stay the same after the forbidden transition?
        for (unsigned short u=firstGainWrong+1; u<nTSamples_; u++){
          if( gainWrong == xtalGains_[u]) wrongGainStaysTheSame=true; 
          else                            wrongGainStaysTheSame=false; 
        }// END loop on samples after forbidden transition
      }// if firstGainWrong!=0 && firstGainWrong<8

      if (numGainWrong>0) {

    
        edm::LogWarning("EcalRawToDigiDevGainSwitch")
          <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n A wrong gain transition switch was found in strip "<<stripId<<" and xtal "<<xtalId;    

        (*invalidGainsSwitch_)->push_back(*pDetId_);

         errorOnXtal = true;
      } 

      if(wrongGainStaysTheSame){

        edm::LogWarning("EcalRawToDigiDevGainSwitch")
          <<"\n For event "<<event_->l1A()<<",dcc "<<mapper_->getActiveDCC()<<" and tower "<<towerId_
          <<"\n A wrong gain switch stay was found in strip "<<stripId<<" and xtal "<<xtalId;
      
       (*invalidGainsSwitchStay_)->push_back(*pDetId_);       

        errorOnXtal = true;  
      }

      //Add frame to collection only if all data format and gain rules are respected
      if(errorOnXtal&&frameAdded) {
	(*digis_)->pop_back();
      }

  
//   }// End on check of det id
  
  }  	
  
  //Point to begin of next xtal Block
  data_ += numbDWInXtalBlock_;
}
