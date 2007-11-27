/*
  Author: Adam Hunt
  email:  ahunt@princeton.edu
  Date:   2007-08-24
*/

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include <unistd.h>

using namespace std;

namespace HCAL_HLX{
  TCPReceiver::TCPReceiver(){
    acquireMode = 0;
    servPort = 0;
    servIP = "127.0.0.1";
    Connected = false;
  }
 
  TCPReceiver::TCPReceiver(unsigned short int port, std::string IP, unsigned char mode = 0){
    acquireMode = mode;
    servPort = port;
    servIP = IP;
    Connected = false;
  }

  TCPReceiver::~TCPReceiver(){
    Disconnect();
  }

  int TCPReceiver::ReceiveLumiSection(){
    // cout << "In " << __PRETTY_FUNCTION__ << endl;

    unsigned short int i, j, k;
 
    if(acquireMode == 0){  // real data
      if(Connected == false)
	return 701;
      
      unsigned int bytesRcvd, bytesToReceive, totalBytesRcvd;
      const unsigned int Buffer_Size = 8192;
      char *Buffer;
      char *BigBuffer;
      time_t tempTime, curTime;
      fd_set fds;
      time(&curTime);
           
      bytesToReceive = sizeof(lumiSection);
      Buffer = new char[Buffer_Size];
      BigBuffer = new char[bytesToReceive];
      totalBytesRcvd = 0;

      memset(reinterpret_cast<char *>(&lumiSection), 0, Buffer_Size);
      memset(Buffer, 0, Buffer_Size);
      memset(BigBuffer, 0, bytesToReceive);

      usleep(10000);

      while(totalBytesRcvd < bytesToReceive){

	FD_ZERO(&fds);
	FD_SET(tcpSocket, &fds); // adds sock to the file descriptor set

	if(select(tcpSocket+1, &fds, NULL, NULL, 0)> 0){

	  if (FD_ISSET(tcpSocket, &fds)) {

	    if((bytesRcvd = recv(tcpSocket, Buffer, Buffer_Size, 0))<=0){
	      delete [] BigBuffer;
	      delete [] Buffer;
	      return 501;
	    }else{
	      
	      if((totalBytesRcvd + bytesRcvd)<= bytesToReceive){
		memcpy(&BigBuffer[totalBytesRcvd], Buffer, bytesRcvd);
	      }else{
		cout << "***** MEM OVER FLOW: Did someone forget to update LumiStructures.hh? *****" << endl;
		delete [] BigBuffer;
		delete [] Buffer;
		return 502;
	      }
	      totalBytesRcvd += bytesRcvd;
	    }
	  }
	}
      }
      
      memcpy(reinterpret_cast<char *>(&lumiSection), BigBuffer, sizeof(lumiSection));
      delete [] Buffer;
      delete [] BigBuffer;

      return 1; // success
    }
  
    if(acquireMode == 1){ // fill with fake data. Should be unique.
      lumiSection.hdr.runNumber = 1;
      lumiSection.hdr.startOrbit = 2;
      lumiSection.hdr.numOrbits = 3;
      lumiSection.hdr.numBunches = 4;
      lumiSection.hdr.numHLXs = 5;
      lumiSection.hdr.bCMSLive = true;
      lumiSection.hdr.sectionNumber = 120;
       
      lumiSection.lumiSummary.DeadtimeNormalization = 7;
      lumiSection.lumiSummary.LHCNormalization = 8;
      lumiSection.lumiSummary.InstantLumi = 9;
      lumiSection.lumiSummary.InstantLumiErr = 10;
      lumiSection.lumiSummary.InstantLumiQlty = 11;
      lumiSection.lumiSummary.InstantETLumi = 12;
      lumiSection.lumiSummary.InstantETLumiErr = 13;
      lumiSection.lumiSummary.InstantETLumiQlty = 14;
      lumiSection.lumiSummary.InstantOccLumi[0] = 15;
      lumiSection.lumiSummary.InstantOccLumiErr[0] = 16;
      lumiSection.lumiSummary.InstantOccLumiQlty[0] = 17;
      lumiSection.lumiSummary.lumiNoise[0] = 18;
      lumiSection.lumiSummary.InstantOccLumi[1] = 19;
      lumiSection.lumiSummary.InstantOccLumiErr[1] = 20;
      lumiSection.lumiSummary.InstantOccLumiQlty[1] = 21;
      lumiSection.lumiSummary.lumiNoise[1] = 22;

      for(j=0; j < 3564; j++){
	//lumiSection.lumiDetail.LHCLumi[j] = 1*j;
	lumiSection.lumiDetail.ETLumi[j] = 2*j;
	lumiSection.lumiDetail.ETLumiErr[j] = 3*j;
	lumiSection.lumiDetail.ETLumiQlty[j] = 4*j;
	lumiSection.lumiDetail.OccLumi[0][j] = 5*j;
	lumiSection.lumiDetail.OccLumiErr[0][j] = 6*j;
	lumiSection.lumiDetail.OccLumiQlty[0][j] = 7*j;
	lumiSection.lumiDetail.OccLumi[1][j] = 8*j;
	lumiSection.lumiDetail.OccLumiErr[1][j] = 9*j;
	lumiSection.lumiDetail.OccLumiQlty[1][j] = 10*j;
      }

      for(i=0; i<36; i ++){
	lumiSection.etSum[i].hdr.numNibbles = 7;
	lumiSection.occupancy[i].hdr.numNibbles = 8;
	lumiSection.lhc[i].hdr.numNibbles = 9;
      
	lumiSection.etSum[i].hdr.bIsComplete = true;
	lumiSection.occupancy[i].hdr.bIsComplete = true;
	lumiSection.lhc[i].hdr.bIsComplete = true;
      
	for(j=0; j < 3564; j ++){
	  lumiSection.etSum[i].data[j] = 6*j+ 10*i;
	  for(k=0; k < 6; k++){
	    lumiSection.occupancy[i].data[k][j] = k*j + 11*i;
	  }
	  lumiSection.lhc[i].data[j]= 7*j + 12*i;
	}
      }
      return 1;
    }
  
    if(acquireMode == 2){ // fill with random fake data.
      srand(time(NULL));
      lumiSection.hdr.runNumber = (rand() % 100);
      lumiSection.hdr.startOrbit = (rand() % 100);
      lumiSection.hdr.numOrbits = (rand() % 100);
      lumiSection.hdr.numBunches = (rand() % 100);
      lumiSection.hdr.numHLXs = (rand() % 100);
      lumiSection.hdr.bCMSLive = true;
      lumiSection.hdr.sectionNumber = (rand() %100);

      lumiSection.lumiSummary.DeadtimeNormalization = (rand() % 100);
      lumiSection.lumiSummary.LHCNormalization = (rand() % 100);
      lumiSection.lumiSummary.InstantLumi = (rand() % 100);
      lumiSection.lumiSummary.InstantLumiErr = (rand() % 100);
      lumiSection.lumiSummary.InstantLumiQlty = (rand() % 100);
      lumiSection.lumiSummary.InstantETLumi = (rand() % 100);
      lumiSection.lumiSummary.InstantETLumiErr = (rand() % 100);
      lumiSection.lumiSummary.InstantETLumiQlty = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumi[0] = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumiErr[0] = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumiQlty[0] = (rand() % 100);
      lumiSection.lumiSummary.lumiNoise[0] = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumi[1] = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumiErr[1] = (rand() % 100);
      lumiSection.lumiSummary.InstantOccLumiQlty[1] = (rand() % 100);
      lumiSection.lumiSummary.lumiNoise[1] = (rand() % 100);
      
      for(j=0; j < 3564; j++){
	//lumiSection.lumiDetail.LHCLumi[j] = (rand() % 100);
	lumiSection.lumiDetail.ETLumi[j] = (rand() % 100);
	lumiSection.lumiDetail.ETLumiErr[j] = (rand() % 100);
	lumiSection.lumiDetail.ETLumiQlty[j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumi[0][j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumiErr[0][j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumiQlty[0][j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumi[1][j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumiErr[1][j] = (rand() % 100);
	lumiSection.lumiDetail.OccLumiQlty[1][j] = (rand() % 100);
      }

      for(i=0; i<36; i ++){
	lumiSection.etSum[i].hdr.numNibbles = (rand() % 100);
	lumiSection.occupancy[i].hdr.numNibbles = 8*(rand() % 100);
	lumiSection.lhc[i].hdr.numNibbles = 9*(rand() % 100);
      
	lumiSection.etSum[i].hdr.bIsComplete = true;
	lumiSection.occupancy[i].hdr.bIsComplete = true;
	lumiSection.lhc[i].hdr.bIsComplete = true;
      
	for(j=0; j < 3564; j ++){
	  lumiSection.etSum[i].data[j] = 6*(rand() % 3564);
	  for(k=0; k < 6; k++){
	    lumiSection.occupancy[i].data[k][j] = k*(rand() % 3564);
	  }
	  lumiSection.lhc[i].data[j]= 7*(rand() % 3564);
	}
      }
      return 1;
    }
    return 201;
  }

  bool TCPReceiver::IsConnected(){

    return Connected;
  }
  
  int TCPReceiver::SetPort(unsigned short int port){
    //cout << "In " << __PRETTY_FUNCTION__ << endl;
   
    if(port < 1024)
      return 101;
    servPort = port;
    return 1;
  }
  
  int TCPReceiver::SetMode(unsigned char mode){
    // cout << "In " << __PRETTY_FUNCTION__ << endl;
  
    if(mode > 2)
      return 201;
    acquireMode = mode;
    return 1;
  }

  void TCPReceiver::SetIP(std::string IP){

    servIP = IP;

  }
  
  int TCPReceiver::Connect(){
    //  cout << "In " << __PRETTY_FUNCTION__ << endl;
  
    if(acquireMode == 0){
      if(servPort < 1024)
	return 101;

      //    cout << "Requesting connect" << endl;
      if((tcpSocket = socket(PF_INET, SOCK_STREAM, 0))<0)
	return 301;

      memset(&servAddr, 0, sizeof(servAddr)); 
      servAddr.sin_family = AF_INET;
      servAddr.sin_addr.s_addr = inet_addr(servIP.c_str());
      servAddr.sin_port = htons(servPort);
      // cout << "Connecting" << endl;
      if(connect(tcpSocket, (struct sockaddr *) &servAddr, sizeof(servAddr))<0)
	return 302;

      // cout << "Connected" << endl;
      Connected = true;
      // cout << "In " << __PRETTY_FUNCTION__ << endl;
      //cout << "Connected = " << Connected << endl;

      return 1;
    }
    if(acquireMode == 1)
      return 1;

    if(acquireMode == 2)
      return 1;

    return 201;
  }
  
  int TCPReceiver::Disconnect(){
    // cout << "In " << __PRETTY_FUNCTION__ << endl;
    // cout << "Connected = " << Connected << endl;
    if(Connected){
      //cout << "Shutting down socket" << endl;
      if(shutdown(tcpSocket,SHUT_RDWR)<0)
	return 601;
      
      // cout << "Socket closed" << endl;
      Connected = false;
      return 1;
    }
    //cout << "Never called" << endl;
    return 401;
  }

}
