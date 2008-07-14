// -*- C++ -*-
//
// Package:    SiPixelFedFillerWordEventNumber
// Class:      SiPixelFedFillerWordEventNumber
// 
/**\class SiPixelFedFillerWordEventNumber SiPixelFedFillerWordEventNumber.cc FedFillerWords/SiPixelFedFillerWordEventNumber/src/SiPixelFedFillerWordEventNumber.cc
   
Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Andres Carlos FLOREZ B
//         Created:  Thu Jun 26 09:02:02 CEST 2008
// $Id$
//
//


// system include files
#include <memory>
#include "CalibTracker/SiPixelTools/interface/SiPixelFedFillerWordEventNumber.h"

//======= constructors and destructor
SiPixelFedFillerWordEventNumber::SiPixelFedFillerWordEventNumber(const edm::ParameterSet& iConfig)
{
  SaveFillerWords_bool = iConfig.getParameter<bool>("SaveFillerWords");
  label     =iConfig.getUntrackedParameter<std::string>("InputLabel","source");
  instance = iConfig.getUntrackedParameter<std::string>("InputInstance",""); 
  produces<std::vector<uint32_t> > ("FillerWordEventNumber");
  if (SaveFillerWords_bool == true){
    produces<std::vector<uint32_t> > ("SaveFillerWord");
  }
}

SiPixelFedFillerWordEventNumber::~SiPixelFedFillerWordEventNumber()
{
  
}
//========== Function to decode data words ================================================== 
int SiPixelFedFillerWordEventNumber::PwordSlink64(uint64_t * ldata, const int length, uint32_t &totword) {
  edm::LogInfo("FedFillerWords") <<"Begin of data"<<std::endl;
  
  if( (ldata[0]&0xf000000000000000LL) != 0x5000000000000000LL )//header 
    {     
      return 0;
    }
  
  //========= analyze the data buffer to find private words ================================
  int fif2cnt=0;
  int dumcnt=0;
  int gapcnt=0;
  
  uint32_t gap[8];
  uint32_t dum[8];
  uint32_t word1=0;
  uint32_t word2=0;
  
  uint32_t chan=0;
  uint32_t roc=0;
  
  const uint32_t rocmsk  = 0x3e00000;
  const uint32_t chnlmsk = 0xfc000000;
  
  for(int jk=0;jk<8;jk++)gap[jk]=0;
  for(int jk=0;jk<8;jk++)dum[jk]=0;
  totword=0;
  int fifcnt=1;
  for(int kk=1;kk<length-1;kk++) {
    //======= if statement to make analize just data with the right format ===================   
    if((((ldata[kk]&0xff00000000000000LL)>>32) == 0xa0000000) 
       && (((ldata[kk]&0xffffff00000000LL)>>32)== (uint64_t)(kk+1))){break;}
    
    edm::LogInfo("FedFillerWords") <<kk<<"Data--->0x" <<std::hex<<ldata[kk] <<std::dec<<std::endl;
    
    word2 = (uint32_t) ldata[kk];
    word1 = (uint32_t) (ldata[kk]>>32);
    
    //======= 1st word ======================================================================
    
    chan= ((word1&chnlmsk)>>26);
    roc= ((word1&rocmsk)>>21);
    
    //======count non-error words 
    if(roc<25){
      if(dumcnt>0){
	dumcnt=0;
      }//stale dummy!
      if((chan<5)&&(fifcnt!=1)){
	edm::LogError("FedFillerWords") <<" error in fifo counting!"<<std::endl;
      }
      if((chan>4)&&(chan<10)&&(fifcnt!=2)) {fif2cnt=0;fifcnt=2;}
      if((chan>9)&&(chan<14)&&(fifcnt!=3)) {fif2cnt=0;fifcnt=3;}
      if((chan>13)&&(chan<19)&&(fifcnt!=4)){fif2cnt=0;fifcnt=4;}
      if((chan>18)&&(chan<23)&&(fifcnt!=5)){fif2cnt=0;fifcnt=5;}
      if((chan>22)&&(chan<28)&&(fifcnt!=6)){fif2cnt=0;fifcnt=6;}
      if((chan>27)&&(chan<32)&&(fifcnt!=7)){fif2cnt=0;fifcnt=7;}
      if((chan>31)&&(fifcnt!=8)){fif2cnt=0;fifcnt=8;}
      fif2cnt++;
    }
    //====== Gap Word 
    if(roc==26){
      gap[fifcnt-1]=(0x1000+(word1&0xff));
      gapcnt++;
      vecSaveFillerWords.push_back(word1);
      edm::LogInfo("FedFillerWords") <<"Filler Word---->"
 				     <<std::hex<<word1<<std::dec
 				     <<" ==== Gap   word---->"
 				     <<std::hex<<gap[fifcnt-1]
				     <<std::dec<<std::endl;
      vecFillerWordsEventNumber.push_back(gap[fifcnt-1]);
    }
    //====== Dummy Word 
    if((roc==27)&&((fif2cnt+dumcnt)<6)){
      dumcnt++;
      dum[fifcnt-1]=(0x1000+(word1&0xff));
      vecSaveFillerWords.push_back(word1);
      edm::LogInfo("FedFillerWords") <<"Filler Word---->"
 				     <<std::hex<<word1<<std::dec
 				     <<" ==== Dummy Word---->"
 				     <<std::hex<<dum[fifcnt-1]
				     <<std::dec<<std::endl;
      vecFillerWordsEventNumber.push_back(dum[fifcnt-1]);
    }
    else if((roc==27)&&((fif2cnt+dumcnt)>6)){
      dumcnt=1;
      fif2cnt=0;
      fifcnt++;
    }
    
    //======== 2nd word ============================================================
    
    chan= ((word2&chnlmsk)>>26);
    roc= ((word2&rocmsk)>>21);
    
    if(roc<25)
      {
	if(dumcnt>0){
	  dumcnt=0;
	  edm::LogInfo("FedFillerWords") <<" ***Stale dummy!"<<std::endl;
	}//stale dummy!
	if((chan<5)&&(fifcnt!=1)){
	  edm::LogError("FedFillerWords") <<" error in fifo counting!"<<std::endl;
	}
	if((chan>4)&&(chan<10)&&(fifcnt!=2)) {fif2cnt=0;fifcnt=2;}
	if((chan>9)&&(chan<14)&&(fifcnt!=3)) {fif2cnt=0;fifcnt=3;}
	if((chan>13)&&(chan<19)&&(fifcnt!=4)){fif2cnt=0;fifcnt=4;}
	if((chan>18)&&(chan<23)&&(fifcnt!=5)){fif2cnt=0;fifcnt=5;}
	if((chan>22)&&(chan<28)&&(fifcnt!=6)){fif2cnt=0;fifcnt=6;}
	if((chan>27)&&(chan<32)&&(fifcnt!=7)){fif2cnt=0;fifcnt=7;}
	if((chan>31)&&(fifcnt!=8)){fif2cnt=0;fifcnt=8;}
	fif2cnt++;
      }
    if(roc==26){
      gap[fifcnt-1]=(0x1000+(word2&0xff));
      gapcnt++;
      vecSaveFillerWords.push_back(word2);
      edm::LogInfo("FedFillerWords") <<"Filler Word--->"
 				     <<std::hex<<word2<<std::dec
 				     <<" ==== Gap   Word----->"
 				     <<std::hex<<gap[fifcnt-1]
				     <<std::dec<<std::endl;
      vecFillerWordsEventNumber.push_back(gap[fifcnt-1]);
    }
    if ((roc==27)&&((fif2cnt+dumcnt)<6)){
      dumcnt++;
      dum[fifcnt-1]=(0x1000+(word1&0xff));
      vecSaveFillerWords.push_back(word2);
      edm::LogInfo("FedFillerWords") <<"Filler Word---->"
 				     <<std::hex<<word2<<std::dec
 				     <<" ==== Dummy Word---->"
 				     <<std::hex<<dum[fifcnt-1]
				     <<std::dec<<std::endl;
      vecFillerWordsEventNumber.push_back(dum[fifcnt-1]);
    }
    else if((roc==27)&&((fif2cnt+dumcnt)>6)){
      dumcnt=1;
      fif2cnt=0;
      fifcnt++;
    }
    
    //word check complete
    if(((fif2cnt+dumcnt)==6)&&(dumcnt>0)){ //done with this fifo 
      dumcnt=0;
      fif2cnt=0;
      fifcnt++;
    }
    if((gapcnt>0)&&((dumcnt+fif2cnt)>5)){//done with this fifo
      gapcnt=0;
      fifcnt++;
      fif2cnt=0;
      dumcnt=0;
    }
    else if((gapcnt>0)&&((dumcnt+fif2cnt)<6)){ 
      gapcnt=0;
    }  
    
  }//==End of fifo-3 word loop
  //========== FPGAs Status ==================================================
  status = 0;
  
  if(gap[0]>0) {
    totword=(gap[0]&0xff);
    status=1;
  }
  else if(gap[1]>0){
    totword=(gap[1]&0xff);
    status=1;
  }
  else if(dum[0]>0){
    totword=(dum[0]&0xff);
    status=1;
  }
  else if(dum[1]>0){
    totword=(dum[1]&0xff);
    status=1;
  }
  
  if(gap[2]>0) {
    totword=totword|((gap[2]&0xff)<<8);
    status=status|0x2;
  }
  else if(gap[3]>0){
    totword=totword|((gap[3]&0xff)<<8);
    status=status|0x2;
  }
  else if(dum[2]>0){
    totword=totword|((dum[2]&0xff)<<8);
    status=status|0x2;
  }
  else if(dum[3]>0){
    totword=totword|((dum[3]&0xff)<<8);
    status=status|0x2;
  }
  
  if(gap[4]>0) {
    totword=totword|((gap[4]&0xff)<<16);
    status=status|0x4;
  }
  else if(gap[5]>0){
    totword=totword|((gap[5]&0xff)<<16);
    status=status|0x4;
  }
  else if(dum[4]>0){
    totword=totword|((dum[4]&0xff)<<16);
    status=status|0x4;
  }
  else if(dum[5]>0){
    totword=totword|((dum[5]&0xff)<<16);
    status=status|0x4;
  }
  
  if(gap[6]>0){
    totword=totword|((gap[6]&0xff)<<24);
    status=status|0x8;
  }
  else if(gap[7]>0){
    totword=totword|((gap[7]&0xff)<<24);
    status=status|0x8;
  }
  else if(dum[6]>0){
    totword=totword|((dum[6]&0xff)<<24);
    status=status|0x8;
  }
  else if(dum[7]>0){
    totword=totword|((dum[7]&0xff)<<24);
    status=status|0x8;
  }
  edm::LogInfo("FedFillerWords") <<"total word = 0x"
 				 <<std::hex<<totword
				 <<std::hex<<" Status = 0x"
				 <<status<<std::dec<<std::endl;
  return(status);
  
}

void
SiPixelFedFillerWordEventNumber::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  edm::Handle<FEDRawDataCollection> buffers;  
  iEvent.getByLabel( label, instance, buffers);
  std::auto_ptr<std::vector<uint32_t> > FillerWordEventNumbers(new std::vector<uint32_t>);
  std::auto_ptr<std::vector<uint32_t> > SaveFillerWords(new std::vector<uint32_t>);
  //===== Loop over all the FEDs ========================================================
  FEDNumbering fednum;
  std::pair<int,int> fedIds = fednum.getSiPixelFEDIds();
  fedIds.first  = 0;
  fedIds.second = 39; 
  
  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    edm::LogInfo("FedFillerWords") << " examining FED: " << fedId << std::endl; 
    const FEDRawData& fedRawData = buffers->FEDData( fedId ); //get event data for this fed
    //======== Run the fill word finder...
    if(fedRawData.size()!= 0){
      uint32_t totword;
      int value = PwordSlink64((uint64_t*)fedRawData.data(),(int)fedRawData.size(), totword);
      if(value!=0)
	//====== Verify that the vector is not empty
	if(vecSaveFillerWords.size()!=0){
	  for(vecSaveFillerWords_It = vecSaveFillerWords.begin(); vecSaveFillerWords_It != vecSaveFillerWords.end(); vecSaveFillerWords_It++){
	    SaveFillerWords->push_back(*vecSaveFillerWords_It);
	  }
	}else{
	  edm::LogWarning("FedFillerWords") <<"========= Filler Words Vector is empty! ==========" <<std::endl;
	}
      edm::LogInfo("FedFillerWords") << "Found " << value << " filler words in FED " << fedId << std::endl;
      for (vecFillerWordsEventNumber_It = vecFillerWordsEventNumber.begin(); vecFillerWordsEventNumber_It != vecFillerWordsEventNumber.end(); vecFillerWordsEventNumber_It++){
	FillerWordEventNumbers->push_back(*vecFillerWordsEventNumber_It);
      }
    }
  }
  iEvent.put(FillerWordEventNumbers , "FillerWordEventNumber");
  //====== bool variable to be controled in the config file, allows the user to put or
  //       the filler words inside the output root file
  if(SaveFillerWords_bool == true)
    {
      iEvent.put(SaveFillerWords, "SaveFillerWord");
    }
  vecSaveFillerWords.erase(vecSaveFillerWords.begin(), vecSaveFillerWords.end());    
  vecFillerWordsEventNumber.erase(vecFillerWordsEventNumber.begin(), vecFillerWordsEventNumber.end());
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelFedFillerWordEventNumber::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelFedFillerWordEventNumber::endJob() {
}
