#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"

namespace HCAL_HLX{

  TCPReceiver::TCPReceiver(){
  }

  
  TCPReceiver::TCPReceiver(unsigned short int port = 50002, unsigned char mode = 0){
    aquireMode = mode;
    listenPort = port;
  }

  TCPReceiver::~TCPReceiver(){
    Disconnect();
  }

  int TCPReceiver::ReceiveLumiSection(){
    
    unsigned int i, j, k;

    if(aquireMode == 0){  // real data

      //return 0; // success

      return 1; // failed
    }

    if(aquireMode == 1){ // fill with fake data. Should be unique.
      lumiSection.hdr.runNumber = 1;
      lumiSection.hdr.startOrbit = 2;
      lumiSection.hdr.numOrbits = 3;
      lumiSection.hdr.numBunches = 4;
      lumiSection.hdr.numHLXs = 5;
      lumiSection.hdr.bCMSLive = 6;

      for(i=0; i<36; i ++){
	lumiSection.etSum[i].hdr.numNibbles = 7;
	lumiSection.occupancy[i].hdr.numNibbles = 8;
	lumiSection.lhc[i].hdr.numNibbles = 9;
      
	lumiSection.etSum[i].hdr.bIsComplete = true;
	lumiSection.occupancy[i].hdr.bIsComplete = true;
	lumiSection.lhc[i].hdr.bIsComplete = true;

	for(j=0; j < 4096; j ++){
	  lumiSection.etSum[i].data[j] = 6*j;
	  for(k=0; k < 6; k++){
	    lumiSection.occupancy[i].data[k][j] = k*j;
	  }
	  lumiSection.lhc[i].data[j]= 7*j;
	}
      }
      return 0;
    }


    return 101; // aquireMode invalid
  }

  int TCPReceiver::SetPort(unsigned int port){

    listenPort = port;
    return 0; // success
  }
  
  int TCPReceiver::SetMode(unsigned char mode){
    
    if(mode > 1)
      return 101; // invaild mode
    aquireMode = mode;
    return 0; // success
  }

  int TCPReceiver::Connect(){
    if(aquireMode == 0){
      // do something to connect
      Connected = true;
    }
    if(aquireMode == 1){
      // do nothing to Connect
    }

    return 0; // success
  }

  int TCPReceiver::Disconnect(){
    if(Connected){
      // do something to disconnect
      Connected = false;
    }
    return 0;
  }
}
