#ifndef HLXTCP_H
#define HLXTCP_H

#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

namespace HCAL_HLX{

  class TCPReceiver{

  public:
    TCPReceiver();
    TCPReceiver(unsigned short int, unsigned char);
    ~TCPReceiver();
    int ReceiveLumiSection();
    HCAL_HLX::LUMI_SECTION lumiSection;
    int SetPort(unsigned int);
    int SetMode(unsigned char);
    int Connect();
    int Disconnect();
  private:
    unsigned short int listenPort;
    unsigned char aquireMode;
    bool Connected;
  };

}
#endif
