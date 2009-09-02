/*
  Author:  Adam Hunt
  email:   ahunt@princeton.edu
  Date:    2007-08-25
*/

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include <iostream>
#include <signal.h>
#include <cstdlib>

using std::cout;
using std::endl;

int gContinue=1;

void CtrlC(int aSigNum) {
  std::cout << "Ctrl-c detected, stopping run" << std::endl;
  gContinue=0;
}

int main(){

  HCAL_HLX::TCPReceiver HT;
  HCAL_HLX::LUMI_SECTION L;

  int errorCode;
  
  errorCode = HT.SetMode(0);
  cout << "SetMode: " << errorCode << endl;
  if(errorCode != 1)
    exit(1);
  errorCode = HT.SetPort(50002);
  cout << "SetPort: " << errorCode << endl;
   if(errorCode != 1)
    exit(1);
 
  while(gContinue){
    if(!HT.IsConnected()){
      errorCode = HT.Connect();
      cout << "Connect: " << errorCode << endl;
      if(errorCode != 1)
	exit(1);
    }

    errorCode = HT.ReceiveLumiSection(L);
    cout << "ReceiveLumiSection(): " << errorCode << endl;
    
    HT.VerifyFakeData(L);

    if(errorCode != 1)
      exit(1);
    
  }
  return 0;
}
