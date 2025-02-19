//_________________________________________________________
//
//  CSCTMBMiniScope July 2010  Alexander Sakharov                            
//  Unpacks TMB Logic MiniScope Analyzer and stores in CSCTMBMiniScope.h  
//_________________________________________________________
//


#include "EventFilter/CSCRawToDigi/interface/CSCTMBMiniScope.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

CSCTMBMiniScope::CSCTMBMiniScope(unsigned short *buf,int Line6b07,int Line6E07) {

  size_ = UnpackMiniScope(buf,Line6b07,Line6E07);

} ///CSCTMBMiniScope


int CSCTMBMiniScope::UnpackMiniScope(unsigned short *buf,int Line6b07,int Line6E07) {


  if((Line6E07-Line6b07) != 0) {

    /// Get tbin and tbin before pre-trigger
    miniScopeTbinCount = buf[Line6b07+1] & 0x00FF;
    miniScopeTbinPreTrigger = (buf[Line6b07+1] >> 8) & 0x000F;
    
    LogTrace("CSCTMBMiniScope") << " MiniScope Found | Tbin: " << miniScopeTbinCount <<
                 " | Tbin Pretrigger: " << miniScopeTbinPreTrigger << std::endl;
        
     miniScopeAdress.clear();
     miniScopeData.clear();

     for(int i=0; i<miniScopeTbinCount; i++){
        miniScopeAdress.push_back(284+i);
        miniScopeData.push_back(buf[Line6b07 + 1+i]);
     }
     
     //print();
  } ///end if((Line6E07-Line6b07)


  return (Line6E07-Line6b07 + 1);

} ///UnpackScope

std::vector<int> CSCTMBMiniScope::getChannelsInTbin(int data) const {
                 std::vector<int> channelInTbin;
                 channelInTbin.clear();
                 for(int k=0; k<14; k++){
                              int chBit=0;
                              chBit = (data >> k) & 0x1;
                              if(chBit !=0)
                              channelInTbin.push_back(k);
        }
        return channelInTbin;
}


void CSCTMBMiniScope::print() const {
     for(unsigned int k=0; k<getAdr().size();++k){
           if(k==0){
             std::cout << " Adr = " << getAdr()[k] << " | Data: " 
                                     << std::hex <<  getData()[k] << std::dec << std::endl;
           }
           else{
           std::cout << " Adr = " << getAdr()[k] << " | Data: " 
                                     << std::hex <<  getData()[k] << std::dec << " ==>| Ch# ";
                                      for(unsigned int j=0; j<getChannelsInTbin(getData()[k]).size(); j++){
                                         std::cout << " " << getChannelsInTbin(getData()[k])[j];
                                      }
                                     std::cout << std::endl;
           }
           }
}
