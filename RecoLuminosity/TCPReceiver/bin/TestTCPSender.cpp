/*
  Author: Adam Hunt
  email:  ahunt@princeton.edu
  Date:   2007-08-24
*/

#include <arpa/inet.h>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

int gContinue = 1;
bool Connected = false;
char *Buffer;

void CtrlC(int aSigNum) {
  std::cout << "Ctrl-c detected, stopping run" << std::endl;
  gContinue = 0;
  delete[] Buffer;
  exit(1);
}

int main() {
  using std::cout;
  using std::endl;

  int servSock;
  int clntSock = -1;

  unsigned int runCount = 1;
  unsigned int orbitCount = 0;
  unsigned int sectionCount = 0;
  unsigned int sizeOfRun = 960;
  unsigned int sizeOfSection = 1000;

  int er;

  HCAL_HLX::LUMI_SECTION lumiSection;
  HCAL_HLX::TCPReceiver HT;

  struct sockaddr_in servAddr;
  struct sockaddr_in clntAddr;
  unsigned int clntLen;
  unsigned short servPort = 51006;
  unsigned int Buffer_Size;

  signal(SIGINT, CtrlC);

  clntLen = sizeof(clntAddr);
  Buffer_Size = sizeof(lumiSection);
  Buffer = new char[Buffer_Size];

  if ((servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
    cout << "***  socket failed **** " << endl;
    exit(1);
  }

  memset(&servAddr, 0, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servAddr.sin_port = htons(servPort);
  do {
    if ((er = bind(servSock, (struct sockaddr *)&servAddr, sizeof(servAddr))) < 0) {
      cout << " *** bind failed *** " << endl;
      cout << " Attempting to bind in 30 seconds " << endl;
      sleep(30);
    }
  } while (er < 0 && gContinue);

  if (listen(servSock, 5) < 0) {
    cout << " *** listen failed *** " << endl;
    exit(1);
  }
  memset(reinterpret_cast<char *>(&lumiSection), 0, Buffer_Size);
  memset(Buffer, 0, Buffer_Size);

  while (gContinue) {
    cout << " ** Generating Lumi Section ** " << endl;
    HT.GenerateFakeData(lumiSection);

    orbitCount += 4;

    if (orbitCount >= sizeOfSection) {
      sectionCount++;
      orbitCount = 0;
    }

    if (sectionCount >= sizeOfRun) {
      runCount++;
      sectionCount = 0;
    }

    lumiSection.hdr.runNumber = runCount;
    lumiSection.hdr.sectionNumber = sectionCount;
    lumiSection.hdr.startOrbit = orbitCount - 4;
    lumiSection.hdr.numOrbits = 4;
    lumiSection.hdr.numBunches = 3546;
    lumiSection.hdr.numHLXs = 36;
    lumiSection.hdr.bCMSLive = true;

    memcpy(Buffer, reinterpret_cast<char *>(&lumiSection), Buffer_Size);

    if (Connected == false) {
      do {
        cout << " **** Waiting *** " << endl;
        if ((clntSock = accept(servSock, (struct sockaddr *)&clntAddr, &clntLen)) < 0)
          cout << " ** accept() failed ** " << endl;
      } while (clntSock < 0 && gContinue);
      Connected = true;

    } else {
      cout << " ** Sending Lumi Section - Run: " << lumiSection.hdr.runNumber
           << " Section: " << lumiSection.hdr.sectionNumber << " Orbit: " << lumiSection.hdr.startOrbit << " ** "
           << endl;
      if (send(clntSock, Buffer, Buffer_Size, 0) != (int)Buffer_Size) {
        cout << " ** send failed ** " << endl;
        Connected = false;
      }
      sleep(3);
    }
  }
}
