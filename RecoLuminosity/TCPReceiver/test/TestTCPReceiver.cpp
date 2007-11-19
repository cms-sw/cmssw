/*
  Author:  Adam Hunt
  email:   ahunt@princeton.edu
  Date:    2007-08-25
*/

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include <iostream>
#include <signal.h>


using std::cout;
using std::endl;

int gContinue=1;

void CtrlC(int aSigNum) {
  std::cout << "Ctrl-c detected, stopping run" << std::endl;
  gContinue=0;
}

int main(){
  unsigned short i, j, k;
  HCAL_HLX::TCPReceiver HT;
  int errorCode;
  unsigned int ErrorCount = 0;
  
  errorCode = HT.SetMode(0);
  cout << "SetMode: " << errorCode << endl;
  if(errorCode != 1)
    exit(1);
  errorCode = HT.SetPort(50002);
  cout << "SetPort: " << errorCode << endl;
   if(errorCode != 1)
    exit(1);
 
  while(gContinue){
    
    errorCode = HT.Connect();
    cout << "Connect: " << errorCode << endl;
  if(errorCode != 1)
    exit(1);
  
    errorCode = HT.ReceiveLumiSection();
    cout << "ReceiveLumiSection(): " << errorCode << endl;
  if(errorCode != 1)
    exit(1);

    if( HT.lumiSection.hdr.runNumber != 1){
      cout << "Error -5 " << endl;
      ErrorCount++;
    }
    if(HT.lumiSection.hdr.startOrbit != 2){
      cout << "Error -4" << endl;
      ErrorCount++;
    }
    if(HT.lumiSection.hdr.numOrbits != 3){
      cout << "Error -3" << endl;
      ErrorCount++;
    }
    if(HT.lumiSection.hdr.numBunches != 4){
      cout << "Error -2" << endl;
      ErrorCount++;
    }
    if(HT.lumiSection.hdr.numHLXs != 5){
      cout << "Error -1" << endl;
      ErrorCount++;
    }
    if(HT.lumiSection.hdr.bCMSLive != true){
      cout << "Error 0" << endl;
      ErrorCount++;
    }
  
    for(i=0; i<36; i ++){
      if(HT.lumiSection.etSum[i].hdr.numNibbles != 7){
	cout << "Error 1 " << i << endl;
	ErrorCount++;
      }
      if(HT.lumiSection.occupancy[i].hdr.numNibbles != 8){
	cout << "Error 2 " << i << endl;
	ErrorCount++;
      }
      if(HT.lumiSection.lhc[i].hdr.numNibbles != 9){
	cout << "Error 3" << i << endl;
	ErrorCount++;
      } 
      if(HT.lumiSection.etSum[i].hdr.bIsComplete != true){
	cout << "Error 4 " << i << endl;
	ErrorCount++;
      }
      if(HT.lumiSection.occupancy[i].hdr.bIsComplete != true){
	cout << "Error 5 " << i << endl;
	ErrorCount++;
      }
      if(HT.lumiSection.lhc[i].hdr.bIsComplete != true){
	cout << "Error 6 " << i << endl;
	ErrorCount++;
      }
    
      for(j=0; j < 3564; j ++){
	if(HT.lumiSection.etSum[i].data[j] != 6*j){
	  cout << "Error 7 " << i << ":" << j << endl;
	  ErrorCount++;
	}
	for(k=0; k < 6; k++){
	  if(HT.lumiSection.occupancy[i].data[k][j] != k*j){
	    cout << "Error 8 " << i << ":" << j << ":" << k << endl;
	    ErrorCount++;
	  }
	}
	if(HT.lumiSection.lhc[i].data[j] != 7*j){
	  cout << "Error 9 " << i << ":" << j << endl;
	  ErrorCount++;
	}
      }
    }

    errorCode = HT.Disconnect();
    cout << "Disconnect: " << errorCode << endl;
   if(errorCode != 1)
    exit(1);

   cout << ErrorCount << endl;
  }
  return 0;
}
