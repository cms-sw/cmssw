// -*- C++ -*-
//
// Package:    SiPixelFedFillerWordEventNumber 
// Class:      SiPixelFedFillerWordEventNumber 
// 
/**\class SiPixelFedFillerWordEventNumber  SiPixelFedFillerWordEventNumber .cc FedFillerWords/SiPixelFedFillerWordEventNumber /src/SiPixelFedFillerWordEventNumber .cc
   
Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Andres Carlos FLOREZ B
//         Created:  Thu Jun 26 09:02:02 CEST 2008
// $Id: SiPixelFedFillerWordEventNumber.cc,v 1.6 2011/07/01 07:05:03 eulisse Exp $
//
//


// system include files
#include <memory>
#include "SiPixelFedFillerWordEventNumber.h"

//======= constructors and destructor
SiPixelFedFillerWordEventNumber ::SiPixelFedFillerWordEventNumber (const edm::ParameterSet& iConfig)
{
  SaveFillerWordsbool = iConfig.getParameter<bool>("SaveFillerWords");
  label     =iConfig.getUntrackedParameter<std::string>("InputLabel","source");
  instance = iConfig.getUntrackedParameter<std::string>("InputInstance",""); 
  produces<std::vector<uint32_t> > ("FillerWordEventNumber1");
  produces<std::vector<uint32_t> > ("FillerWordEventNumber2");
  if (SaveFillerWordsbool == true){
    produces<std::vector<uint32_t> > ("SaveFillerWord");
  }
}

SiPixelFedFillerWordEventNumber ::~SiPixelFedFillerWordEventNumber ()
{
  
}
unsigned int SiPixelFedFillerWordEventNumber ::CalibStatFillWord(unsigned int totword, int status){
  //===== Variables to get each filler word out of the totword and 
  //      to conform the last 16 bit filler word if Filler3 is zero.   
  unsigned int Filler1      = (totword)&0x000000ff;
  unsigned int Filler2      = ((totword)&0x0000ff00)>>8;
  unsigned int Filler3      = ((totword)&0x00ff0000)>>16;
  unsigned int Filler4      = ((totword)&0xffff0000)>>16;
  unsigned int maskFiller4  = ((totword)&0xff000000)>>16;
  unsigned int Filler14     = (Filler1&maskFiller4);
  unsigned int Filler24     = (Filler2&maskFiller4);
  unsigned int CalibFiller1 = 0;
  unsigned int CalibFiller2 = 0;
  bool BoolStat = false;
  //=====Possible cases for the totword. "CalibFiller1" and "CalibFiller2" take their valur
  //     according to the value of the totword.
  if ((status == 0x1)||(status == 0x9)){
    CalibFiller1 = Filler1;
    if((status == 0x9)){CalibFiller2 = Filler14;}
  }
  if ((status == 0x2)||(status == 0xa)){
    CalibFiller1 = Filler2;
    if((status == 0xa)){CalibFiller2 = Filler24;}
  }
  if ((status == 0x4)||(status == 0xc)){
    CalibFiller1 = Filler3;
    if((status == 0xc)){CalibFiller2 = Filler4;}
  }
  if ((status == 0x8)){CalibFiller2 = Filler4;}  
  if((status == 0x7)||(status == 0xf)){
    if((Filler1 == Filler2)&&(Filler1 == Filler3)&&(Filler2 == Filler3)){
      CalibFiller1 = Filler1;
      BoolStat = true;
      if(status == 0xf){CalibFiller2 = Filler4;} 
    }else{
      edm::LogError("AnazrFedFillerWords")<<"Status: "<<status << "Event ID in Filler words don't match"
                                          <<'\t'<<"Filler1: "<<(Filler1%256)
					  <<'\t'<<"Filler2: "<<(Filler2%256)
					  <<'\t'<<"Filler3: "<<(Filler3%256)
					  <<std::endl; 
    }
  }
  if((status == 0x3)||(status == 0xb)){
    if((Filler1 == Filler2)){
      CalibFiller1 = Filler1;
      BoolStat = true;
      if(status == 0xb){CalibFiller2 = Filler14;}
    }else{
      edm::LogError("AnazrFedFillerWords")<<"Status: "<<status << "Event ID in Filler words don't match"
                                          <<'\t'<<"Filler1: "<<(Filler1%256)
					  <<'\t'<<"Filler2: "<<(Filler2%256)
					  <<std::endl;
    }
  }
  if((status == 0x5)||(status == 0xd)){
    if((Filler1 == Filler3)){
      CalibFiller1 = Filler1;
      BoolStat = true;
      if(status == 0xd){CalibFiller2 = Filler4;}
    }else{
      edm::LogError("AnazrFedFillerWords")<<"Status: "<<status << "Event ID in Filler words don't match"
                                          <<'\t'<<"Filler1: "<<(Filler1%256)
					  <<'\t'<<"Filler3: "<<(Filler3%256) 
                                          <<std::endl;
    }
  }
  if((status == 0x6)||(status == 0xe)){
    if((Filler2 == Filler3)){
      CalibFiller1 = Filler2;
      BoolStat = true;
      if(status == 0xe){CalibFiller2 = Filler4;}
    }else{
      edm::LogError("AnazrFedFillerWords")<<"Status: "<<status << "Event ID Filler words don't match"
                                          <<'\t'<<"Filler2: "<<(Filler2%256)
					  <<'\t'<<"Filler3: "<<(Filler3%256) 
                                          <<std::endl;
    }
  }
  //===== Using the Event number from CMSSW to get a value to compare with the value encoded 
  //      in the filler words.
  unsigned int CalibEvtNum = ((EventNum -1)/10);
  if((CalibFiller1 != 0)&&(CalibEvtNum != CalibFiller1)){
    edm::LogError("AnazrFedFillerWords")<<"Error, Event ID Numbers Don't match---->"<<"Filler1 Event ID: "
                                        << CalibFiller1 <<'\t'<<"Run Event ID: "
					<<CalibEvtNum<<'\t'<<std::endl;
  }else if((CalibFiller1 != 0)&&(CalibEvtNum == CalibFiller1)){
    vecFillerWordsEventNumber1.push_back((CalibFiller1%256));
    edm::LogInfo("AnazrFedFillerWords")<<"Filler1 Event ID: "
                                       <<(CalibFiller1%256)<<std::endl;
  }else if((CalibFiller2 != 0)&&(BoolStat == true)){
    vecFillerWordsEventNumber2.push_back((((CalibFiller2%65536)&(0xff00))>>8));
    edm::LogInfo("AnazrFedFillerWords")<<"Filler2 Event ID:"
                                       <<(((CalibFiller2%65536)&(0xff00))>>8)<<std::endl;
  }else if((CalibFiller2 != 0)&&(BoolStat == false)){
    if((status == 0x9)||(status == 0xa)||(status == 0xc)){
      vecFillerWordsEventNumber2.push_back((((CalibFiller2%65536)&(0xff00))>>8));
      edm::LogInfo("AnazrFedFillerWords")<<"Filler2 Event ID:"<<(((CalibFiller2%65536)&(0xff00))>>8)<<std::endl;
    }else if((status == 0x8)){
      edm::LogError("AnazrFedFillerWords")<<"Status: "<<status 
                                          <<" No Filler1 found, is not possible get any Event ID Number"
                                          <<std::endl;
    }
  }
  
  return 0;
} 
//========== Function to decode data words ================================================== 
int SiPixelFedFillerWordEventNumber ::PwordSlink64(uint64_t * ldata, const int length, uint32_t &totword) {
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
    }
    //====== Dummy Word 
    if((roc==27)&&((fif2cnt+dumcnt)<6)){
      dum[fifcnt-1]=(0x1000+(word1&0xff));
      dumcnt++;
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
    }
    if ((roc==27)&&((fif2cnt+dumcnt)<6)){
      dum[fifcnt-1]=(0x1000+(word1&0xff));
      dumcnt++;
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
  vecSaveFillerWords.push_back(totword);
  if((EventNum%10) == 0){
  CalibStatFill = CalibStatFillWord(totword, status);
  }
  edm::LogInfo("FedFillerWords") <<"total word = 0x"
 				 <<std::hex<<totword
				 <<std::hex<<" Status = 0x"
				 <<status<<std::dec<<std::endl;
  return(status);
  
}

void
SiPixelFedFillerWordEventNumber ::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  EventNum = iEvent.id().event();
  edm::Handle<FEDRawDataCollection> buffers;  
  iEvent.getByLabel( label, instance, buffers);
  std::auto_ptr<std::vector<uint32_t> > FillerWordEventNumbers1(new std::vector<uint32_t>);
  std::auto_ptr<std::vector<uint32_t> > FillerWordEventNumbers2(new std::vector<uint32_t>);
  std::auto_ptr<std::vector<uint32_t> > SaveFillerWords(new std::vector<uint32_t>);
  //===== Loop over all the FEDs ========================================================
  FEDNumbering fednum;
  std::pair<int,int> fedIds;
  fedIds.first  = 0;
  fedIds.second = 39; 
  
  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    edm::LogInfo("FedFillerWords") << " examining FED: " << fedId << std::endl; 
    const FEDRawData& fedRawData = buffers->FEDData( fedId ); //get event data for this fed
    //======== Run the fill word finder...
    if(fedRawData.size()!= 0){
      uint32_t totword;
      int value = PwordSlink64((uint64_t*)fedRawData.data(),(int)fedRawData.size(), totword);
      if(value!=0){
	//====== Verify that the vector is not empty
	if(vecSaveFillerWords.size()!=0){
	  for(vecSaveFillerWords_It = vecSaveFillerWords.begin(); vecSaveFillerWords_It != vecSaveFillerWords.end(); vecSaveFillerWords_It++){
	    SaveFillerWords->push_back(*vecSaveFillerWords_It);
	  }
	}else{
	  edm::LogWarning("FedFillerWords") <<"========= Filler Words Vector is empty! ==========" <<std::endl;
	}
      }
      edm::LogInfo("FedFillerWords") << "Found " << value << " filler words in FED " << fedId << std::endl;
      for (vecFillerWordsEventNumber1_It = vecFillerWordsEventNumber1.begin(); vecFillerWordsEventNumber1_It != vecFillerWordsEventNumber1.end(); vecFillerWordsEventNumber1_It++){
	FillerWordEventNumbers1->push_back(*vecFillerWordsEventNumber1_It);
      }
      for(vecFillerWordsEventNumber2_It = vecFillerWordsEventNumber2.begin(); vecFillerWordsEventNumber2_It != vecFillerWordsEventNumber2.end(); vecFillerWordsEventNumber2_It++){
        FillerWordEventNumbers2->push_back(*vecFillerWordsEventNumber2_It);
      }
    }
  }
  iEvent.put(FillerWordEventNumbers1 , "FillerWordEventNumber1");
  iEvent.put(FillerWordEventNumbers2 , "FillerWordEventNumber2");
  //====== bool variable to be controled in the config file, allows the user to put or
  //       the filler words inside the output root file
  if(SaveFillerWordsbool == true)
    {
      iEvent.put(SaveFillerWords, "SaveFillerWord");
    }
  vecSaveFillerWords.erase(vecSaveFillerWords.begin(), vecSaveFillerWords.end());    
  vecFillerWordsEventNumber1.erase(vecFillerWordsEventNumber1.begin(), vecFillerWordsEventNumber1.end());
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelFedFillerWordEventNumber ::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelFedFillerWordEventNumber ::endJob() {
}
//===== define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedFillerWordEventNumber );
