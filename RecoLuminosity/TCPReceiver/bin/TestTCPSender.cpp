/*
  Author: Adam Hunt
  email:  ahunt@princeton.edu
  Date:   2007-08-24
*/

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <signal.h>
#include <unistd.h>

#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

int gContinue=1;
bool Connected = false;
char * Buffer;

void CtrlC(int aSigNum) {
  std::cout << "Ctrl-c detected, stopping run" << std::endl;
  gContinue=0;
  delete [] Buffer;
  exit(1);
}

int main(){
  using std::cout;
  using std::endl;

  int servSock;
  int clntSock;

  int er;
  unsigned short i, j, k;
  HCAL_HLX::LUMI_SECTION lumiSection;

  struct sockaddr_in servAddr;
  struct sockaddr_in clntAddr;
  unsigned int clntLen;
  unsigned short servPort = 51001;

  unsigned int Buffer_Size;

  signal(SIGINT,CtrlC);  

  clntLen = sizeof(clntAddr);    
  Buffer_Size = sizeof(lumiSection);
  Buffer = new char[Buffer_Size];
  
  if((servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP))<0){
    cout << "***  socket failed **** " << endl;
    exit(1);
  }
  
  memset(&servAddr, 0, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servAddr.sin_port = htons(servPort);
  do{
    if((er = bind(servSock, (struct sockaddr *) &servAddr, sizeof(servAddr))) < 0){
      cout << " *** bind failed *** " << endl;
      cout << " Attempting to bind in 30 seconds " << endl;
      sleep(30);
    }
  }while(er < 0 && gContinue);

  if(listen(servSock, 5) < 0){
    cout << " *** listen failed *** " << endl;
    exit(1);
  }
  memset(reinterpret_cast<char *>(&lumiSection), 0, Buffer_Size);
  memset(Buffer, 0, Buffer_Size); 

  while(gContinue){
    // cout << " ** Writing Lumi Section ** " << endl;
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
      
      for(j=0; j < 3564; j ++){
	lumiSection.etSum[i].data[j] = 6*j;
	for(k=0; k < 6; k++){
	  lumiSection.occupancy[i].data[k][j] = k*j;
	}
	lumiSection.lhc[i].data[j]= 7*j;
      }
    }

    memcpy(Buffer,reinterpret_cast<char *>(&lumiSection), Buffer_Size);

    if(Connected == false){

      do{
	cout << " **** Waiting *** " << endl;
	if((clntSock = accept(servSock, (struct sockaddr *) &clntAddr, &clntLen)) < 0)
	  cout << " ** accept() failed ** " << endl;
      }while(clntSock < 0 && gContinue);
      Connected = true;
      
    } else {
      
      cout << " ** Sending lumi section ** " << endl;
      if(send(clntSock, Buffer, Buffer_Size, 0) != (int)Buffer_Size){
	cout << " ** send failed ** " << endl;
	Connected = false;
      }
      sleep(3);
    }
  }
  
}
