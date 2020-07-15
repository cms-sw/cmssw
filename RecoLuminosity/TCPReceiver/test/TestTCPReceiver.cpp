/*
  Author:  Adam Hunt
  email:   ahunt@princeton.edu
  Date:    2007-08-25
*/

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include <csignal>
#include <cstdlib>
#include <iostream>

int gContinue = 1;

void CtrlC(int aSigNum) {
  std::cout << "Ctrl-c detected, stopping run" << std::endl;
  gContinue = 0;
}

int main() {
  HCAL_HLX::TCPReceiver HT;
  HCAL_HLX::LUMI_SECTION L;

  int errorCode;

  errorCode = HT.SetMode(0);
  std::cout << "SetMode: " << errorCode << std::endl;
  if (errorCode != 1)
    exit(1);
  errorCode = HT.SetPort(50002);
  std::cout << "SetPort: " << errorCode << std::endl;
  if (errorCode != 1)
    exit(1);

  while (gContinue) {
    if (!HT.IsConnected()) {
      errorCode = HT.Connect();
      std::cout << "Connect: " << errorCode << std::endl;
      if (errorCode != 1)
        exit(1);
    }

    errorCode = HT.ReceiveLumiSection(L);
    std::cout << "ReceiveLumiSection(): " << errorCode << std::endl;

    HT.VerifyFakeData(L);

    if (errorCode != 1)
      exit(1);
  }
  return 0;
}
