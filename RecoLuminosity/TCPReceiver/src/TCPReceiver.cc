
/*
  Author: Adam Hunt
  email:  ahunt@princeton.edu
  Date:   2007-08-24
*/

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include <iostream>

#include <unistd.h>
#include <sys/time.h>

// srand rand
#include <cstdlib>

// perror
#include <cstdio>

// tcp
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>

#include <cstring>

namespace HCAL_HLX {

  TCPReceiver::TCPReceiver() {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    acquireMode = 0;
    servPort = 0;
    servIP = "127.0.0.1";
    Connected = false;

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  }

  TCPReceiver::TCPReceiver(unsigned short int port, std::string IP, unsigned char mode = 0) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    acquireMode = mode;
    servPort = port;
    servIP = IP;
    Connected = false;

#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  }

  TCPReceiver::~TCPReceiver() { Disconnect(); }

  void SetupFDSets(fd_set &ReadFDs,
                   fd_set &WriteFDs,
                   fd_set &ExceptFDs,
                   int ListeningSocket = -1,
                   int connectSocket = -1) {  //std::vector & gConnections) {
    FD_ZERO(&ReadFDs);
    FD_ZERO(&WriteFDs);
    FD_ZERO(&ExceptFDs);

    // Add the listener socket to the read and except FD sets, if there
    // is one.
    if (ListeningSocket != -1) {
      FD_SET(ListeningSocket, &ReadFDs);
      FD_SET(ListeningSocket, &ExceptFDs);
    }

    if (connectSocket != -1) {
      FD_SET(connectSocket, &ReadFDs);
      FD_SET(connectSocket, &ExceptFDs);
    }
  }

  int TCPReceiver::ReceiveLumiSection(HCAL_HLX::LUMI_SECTION &localSection) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    int errorCode = 0;

    if (acquireMode == 0) {  // real data
      if (Connected == false) {
        errorCode = 701;  // not connected
      } else {
        unsigned int bytesRcvd, bytesToReceive, totalBytesRcvd;
        const unsigned int Buffer_Size = 8192;
        char *Buffer;
        char *BigBuffer;

        // From John's code

        fd_set fdsRead, fdsWrite, fdsExcept;

        //	int outputcode;
        //int z = 0, localCount = 0;
        time_t tempTime, curTime;
        //int ret;

        time(&curTime);

        bytesToReceive = sizeof(localSection);
        Buffer = new char[Buffer_Size];
        BigBuffer = new char[bytesToReceive];
        totalBytesRcvd = 0;

        memset(reinterpret_cast<char *>(&localSection), 0, Buffer_Size);
        memset(Buffer, 0, Buffer_Size);
        memset(BigBuffer, 0, bytesToReceive);

        usleep(10000);

        while ((totalBytesRcvd < bytesToReceive) && (errorCode == 0)) {
          SetupFDSets(fdsRead, fdsWrite, fdsExcept, -1, tcpSocket);

          if (select(tcpSocket + 1, &fdsRead, nullptr, &fdsExcept, nullptr) > 0) {
            if (FD_ISSET(tcpSocket, &fdsRead)) {
              if ((bytesRcvd = recv(tcpSocket, Buffer, Buffer_Size, 0)) <= 0) {
                perror("Recv Error");
                errorCode = 501;
              } else {
                if ((totalBytesRcvd + bytesRcvd) <= bytesToReceive) {
                  memcpy(&BigBuffer[totalBytesRcvd], Buffer, bytesRcvd);
                } else {
                  std::cout << "***** MEM OVER FLOW: Did someone forget to update LumiStructures.hh? *****"
                            << std::endl;
                  errorCode = 502;
                }
                totalBytesRcvd += bytesRcvd;
                time(&tempTime);
              }
            }
          }
        }

        if (errorCode == 0) {
          memcpy(reinterpret_cast<char *>(&localSection), BigBuffer, sizeof(localSection));
          errorCode = 1;  // success
        }
        delete[] Buffer;
        delete[] BigBuffer;
      }
    } else if (acquireMode == 1) {  // fill with fake data. Should be unique.
      GenerateFakeData(localSection);
      errorCode = 1;
    } else if (acquireMode == 2) {  // fill with random fake data.
      GenerateRandomData(localSection);
      errorCode = 1;
    } else {
      errorCode = 201;
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << " " << errorCode << std::endl;
#endif

    return errorCode;
  }

  bool TCPReceiver::IsConnected() {
#ifdef DEBUG
    std::cout << "Begin and End  " << __PRETTY_FUNCTION__ << " " << Connected << std::endl;
#endif
    return Connected;
  }

  int TCPReceiver::SetPort(unsigned short int port) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    int errorCode;

    if (port < 1024) {
      errorCode = 101;
    } else {
      servPort = port;
      errorCode = 1;
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << " " << errorCode << std::endl;
#endif
    return errorCode;
  }

  int TCPReceiver::SetMode(unsigned char mode) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    int errorCode;

    if (mode > 2) {
      errorCode = 201;
    } else {
      acquireMode = mode;
      errorCode = 1;
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << " " << errorCode << std::endl;
#endif
    return errorCode;
  }

  void TCPReceiver::SetIP(std::string IP) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    servIP = IP;
#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  }

  int TCPReceiver::Connect() {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    int errorCode;

    if (acquireMode == 0) {
      struct hostent *hostInfo = gethostbyname(servIP.c_str());

      if (servPort < 1024) {
        errorCode = 101;  // Protected ports
      } else {
#ifdef DEBUG
        std::cout << "Requesting connection" << std::endl;
#endif
        if ((tcpSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
          perror("Socket Error");
          errorCode = 301;  // Socket failed
        } else {
          memset(&servAddr, 0, sizeof(servAddr));
          servAddr.sin_family = hostInfo->h_addrtype;
          memcpy((char *)&servAddr.sin_addr.s_addr, hostInfo->h_addr_list[0], hostInfo->h_length);
          //  servAddr.sin_addr.s_addr = inet_addr(servIP.c_str());
          servAddr.sin_port = htons(servPort);
#ifdef DEBUG
          std::cout << "Connecting" << std::endl;
#endif
          if (connect(tcpSocket, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0) {
            perror("Connect Error");
            errorCode = 302;  // connect failed
            close(tcpSocket);
          } else {
            Connected = true;
            errorCode = 1;  // Successful connection
          }
        }
      }
    } else if (acquireMode == 1) {
      Connected = true;
      errorCode = 1;  // do nothing for fake data
    } else if (acquireMode == 2) {
      Connected = true;
      errorCode = 1;  // do nothing for random data
    } else {
      errorCode = 201;  // Invalid acquire mode
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << " " << errorCode << std::endl;
#endif
    return errorCode;
  }

  int TCPReceiver::Disconnect() {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    int errorCode = 0;

    if (Connected) {
      if (acquireMode == 0) {
        if (shutdown(tcpSocket, SHUT_RDWR) < 0) {
          perror("Shutdown Error");
          errorCode = 601;  // Disconnect Failed
        } else {
          errorCode = 1;  // Successful Disconnect
        }
      }
      Connected = false;
    } else {
      errorCode = 401;  // Not Connected
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << " " << errorCode << std::endl;
#endif
    return errorCode;
  }

  void TCPReceiver::GenerateFakeData(HCAL_HLX::LUMI_SECTION &localSection) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    int i, j, k;

    localSection.hdr.runNumber = 1;
    localSection.hdr.startOrbit = 2;
    localSection.hdr.numOrbits = 3;
    localSection.hdr.numBunches = 4;
    localSection.hdr.numHLXs = 5;
    localSection.hdr.bCMSLive = true;
    localSection.hdr.sectionNumber = 120;

    timeval tvTemp;
    gettimeofday(&tvTemp, nullptr);
    localSection.hdr.timestamp = tvTemp.tv_sec;
    localSection.hdr.timestamp_micros = tvTemp.tv_usec;

    localSection.lumiSummary.DeadtimeNormalization = 0.7;
    localSection.lumiSummary.LHCNormalization = 0.75;
    localSection.lumiSummary.OccNormalization[0] = 0.8;
    localSection.lumiSummary.OccNormalization[1] = 0.85;
    localSection.lumiSummary.ETNormalization = 0.8;
    localSection.lumiSummary.InstantLumi = 0.9;
    localSection.lumiSummary.InstantLumiErr = 0.10;
    localSection.lumiSummary.InstantLumiQlty = 11;
    localSection.lumiSummary.InstantETLumi = 0.12;
    localSection.lumiSummary.InstantETLumiErr = 0.13;
    localSection.lumiSummary.InstantETLumiQlty = 14;
    localSection.lumiSummary.InstantOccLumi[0] = 0.15;
    localSection.lumiSummary.InstantOccLumiErr[0] = 0.16;
    localSection.lumiSummary.InstantOccLumiQlty[0] = 17;
    localSection.lumiSummary.lumiNoise[0] = 0.18;
    localSection.lumiSummary.InstantOccLumi[1] = 0.19;
    localSection.lumiSummary.InstantOccLumiErr[1] = 0.20;
    localSection.lumiSummary.InstantOccLumiQlty[1] = 21;
    localSection.lumiSummary.lumiNoise[1] = 0.22;

    for (j = 0; j < 3564; j++) {
      localSection.lumiDetail.ETBXNormalization[j] = 0.25 * j / 35640.0;
      localSection.lumiDetail.OccBXNormalization[0][j] = 0.5 * j / 35640.0;
      localSection.lumiDetail.OccBXNormalization[1][j] = 0.75 * j / 35640.0;
      localSection.lumiDetail.LHCLumi[j] = 1 * j / 35640.0;
      localSection.lumiDetail.ETLumi[j] = 2 * j / 35640.0;
      localSection.lumiDetail.ETLumiErr[j] = 3 * j / 35640.0;
      localSection.lumiDetail.ETLumiQlty[j] = 4 * j;
      localSection.lumiDetail.OccLumi[0][j] = 5 * j / 35640.0;
      localSection.lumiDetail.OccLumiErr[0][j] = 6 * j / 35640.0;
      localSection.lumiDetail.OccLumiQlty[0][j] = 7 * j;
      localSection.lumiDetail.OccLumi[1][j] = 8 * j / 35640.0;
      localSection.lumiDetail.OccLumiErr[1][j] = 9 * j / 35640.0;
      localSection.lumiDetail.OccLumiQlty[1][j] = 10 * j;
    }

    for (i = 0; i < 36; i++) {
      localSection.etSum[i].hdr.numNibbles = 7;
      localSection.occupancy[i].hdr.numNibbles = 8;
      localSection.lhc[i].hdr.numNibbles = 9;

      localSection.etSum[i].hdr.bIsComplete = true;
      localSection.occupancy[i].hdr.bIsComplete = true;
      localSection.lhc[i].hdr.bIsComplete = true;

      for (j = 0; j < 3564; j++) {
        localSection.etSum[i].data[j] = 6 * j + 10 * i;
        for (k = 0; k < 6; k++) {
          localSection.occupancy[i].data[k][j] = k * j + 11 * i;
        }
        localSection.lhc[i].data[j] = 7 * j + 12 * i;
      }
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  }

  void TCPReceiver::GenerateRandomData(HCAL_HLX::LUMI_SECTION &localSection) {
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    int i, j, k;

    srand(time(nullptr));
    localSection.hdr.runNumber = 55;  //(rand() % 100);
    localSection.hdr.startOrbit = (rand() % 100);
    localSection.hdr.numOrbits = (rand() % 100);
    localSection.hdr.numBunches = (rand() % 100);
    localSection.hdr.numHLXs = (rand() % 100);
    localSection.hdr.bCMSLive = true;
    localSection.hdr.sectionNumber = (rand() % 100);

    localSection.lumiSummary.DeadtimeNormalization = (float)(rand() % 100) / 100;
    localSection.lumiSummary.LHCNormalization = (float)(rand() % 100) / 100;
    localSection.lumiSummary.OccNormalization[0] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.OccNormalization[1] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.ETNormalization = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantLumi = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantLumiErr = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantLumiQlty = (rand() % 100);
    localSection.lumiSummary.InstantETLumi = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantETLumiErr = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantETLumiQlty = (rand() % 100);
    localSection.lumiSummary.InstantOccLumi[0] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantOccLumiErr[0] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantOccLumiQlty[0] = (rand() % 100);
    localSection.lumiSummary.lumiNoise[0] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantOccLumi[1] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantOccLumiErr[1] = (float)(rand() % 100) / 100;
    localSection.lumiSummary.InstantOccLumiQlty[1] = (rand() % 100);
    localSection.lumiSummary.lumiNoise[1] = (float)(rand() % 100) / 100;

    for (j = 0; j < 3564; j++) {
      localSection.lumiDetail.ETBXNormalization[j] = 0.25 * j / 35640.0;
      localSection.lumiDetail.OccBXNormalization[0][j] = 0.5 * j / 35640.0;
      localSection.lumiDetail.OccBXNormalization[1][j] = 0.75 * j / 35640.0;
      localSection.lumiDetail.LHCLumi[j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.ETLumi[j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.ETLumiErr[j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.ETLumiQlty[j] = (rand() % 100);
      localSection.lumiDetail.OccLumi[0][j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.OccLumiErr[0][j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.OccLumiQlty[0][j] = (rand() % 100);
      localSection.lumiDetail.OccLumi[1][j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.OccLumiErr[1][j] = (float)(rand() % 100) / 100.0;
      localSection.lumiDetail.OccLumiQlty[1][j] = (rand() % 100);
    }

    for (i = 0; i < 36; i++) {
      localSection.etSum[i].hdr.numNibbles = (rand() % 100);
      localSection.occupancy[i].hdr.numNibbles = 8 * (rand() % 100);
      localSection.lhc[i].hdr.numNibbles = 9 * (rand() % 100);

      localSection.etSum[i].hdr.bIsComplete = true;
      localSection.occupancy[i].hdr.bIsComplete = true;
      localSection.lhc[i].hdr.bIsComplete = true;

      for (j = 0; j < 3564; j++) {
        localSection.etSum[i].data[j] = 6 * (rand() % 3564);
        for (k = 0; k < 6; k++) {
          localSection.occupancy[i].data[k][j] = k * (rand() % 3564);
        }
        localSection.lhc[i].data[j] = 7 * (rand() % 3564);
      }
    }

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  }

  bool TCPReceiver::VerifyFakeData(HCAL_HLX::LUMI_SECTION &localSection) {
    HCAL_HLX::LUMI_SECTION L;
    GenerateFakeData(L);
    return !(memcmp(&L, &localSection, sizeof(L)));
  }

}  // namespace HCAL_HLX
