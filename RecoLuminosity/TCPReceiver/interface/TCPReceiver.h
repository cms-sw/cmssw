/*
  Author: Adam Hunt
  email:  ahunt@princeton.edu
  Date:   2007-08-24
*/

#ifndef HLXTCP_H
#define HLXTCP_H

#include <string>

// srand rand
#include <ctime>
#include <cstdlib>

// tcp
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>

// Lumi
#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

/* 
   Error Codes
       0: unknown failure
       1: success
     101: invalid port
        iana.org registers ports from 0 - 1023
     201: invalid mode
        Acceptable modes are 0:  tcp data,  1: constant fake data, 2: random fake data
     301: socket() failed
     302: connect() failed
     401: Disconnect() called without being connected
     501: Failed to Receive Data from server
     601; close() failed

*/
/*
struct ClientConnectionData {
  // Socket identifier
  int sd;

  // Lumi section buffer
  u8 lumiSection[sizeof(LUMI_SECTION)];

  // Amount of data left to write
  u32 leftToWrite;
};
*/
/*
void SetupFDSets(fd_set& ReadFDs, fd_set& WriteFDs,
                 fd_set& ExceptFDs, int ListeningSocket = -1,
                 int connectSocket = -1) { //std::vector & gConnections) {
    FD_ZERO(&ReadFDs);
    FD_ZERO(&WriteFDs);
    FD_ZERO(&ExceptFDs);

     // Add the listener socket to the read and except FD sets, if there
    // is one.
    if (ListeningSocket != -1) {
        FD_SET(ListeningSocket, &ReadFDs);
        FD_SET(ListeningSocket, &ExceptFDs);
    }

    // Add client connections
    */
    /*        std::vector<ClientConnectionData>::iterator it = gConnections.begin();
    while (it != gConnections.end()) {
      if (it->nCharsInBuffer < kBufferSize) {
        // There's space in the read buffer, so pay attention to
        // incoming data.
        FD_SET(it->sd, &ReadFDs);
      }

      //if (it->nCharsInBuffer > 0) {
      // There's data still to be sent on this socket, so we need
      // to be signalled when it becomes writable.
      //FD_SET(it->sd, &WriteFDs);
      //  }

      FD_SET(it->sd, &ExceptFDs);

      ++it;
      }
*/
/*
    if (connectSocket != -1) {
      FD_SET(connectSocket, &ReadFDs);
      FD_SET(connectSocket, &ExceptFDs);
    }
}
*/

namespace HCAL_HLX{

  class TCPReceiver{

  public:
    TCPReceiver();
    TCPReceiver(unsigned short int, std::string,  unsigned char);
    ~TCPReceiver();
    int Connect();
    int SetPort(unsigned short int);
    int SetMode(unsigned char);
    void SetIP(std::string IP);
    int ReceiveLumiSection(HCAL_HLX::LUMI_SECTION & localSection);
    int Disconnect();
    bool IsConnected();    
    bool VerifyFakeData(HCAL_HLX::LUMI_SECTION & localSection);

  private:
    unsigned char acquireMode;
    bool Connected;

    unsigned short servPort;
    std::string servIP;
    int tcpSocket;
    struct sockaddr_in servAddr;
    void GenerateFakeData(HCAL_HLX::LUMI_SECTION & localSection);
    void GenerateRandomData(HCAL_HLX::LUMI_SECTION & localSection);
    //  void SetupFDSets(fd_set& ReadFDs, fd_set& WriteFDs,
    //             fd_set& ExceptFDs, int ListeningSocket = -1,
    //            int connectSocket = -1)
  };

}
#endif
