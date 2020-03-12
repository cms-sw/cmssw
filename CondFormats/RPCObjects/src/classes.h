#include "CondFormats/RPCObjects/src/headers.h"

namespace CondFormats_RPCObjects {
  struct dictionary {
    std::vector<ChamberStripSpec> theStrips;

    std::vector<FebConnectorSpec> theFebs;

    std::vector<LinkBoardSpec> theLBs;

    std::vector<LinkConnSpec> theLinks;

    std::vector<TriggerBoardSpec> theTBs;

    std::map<int, DccSpec> theFeds;

    std::pair<RPCLBLink, RPCFebConnector> theRPCLinkPair;
    std::map<RPCLBLink, RPCFebConnector> theRPCLinkMap;
    std::pair<RPCDCCLink, RPCLBLink> theRPCDCCLinkPair;
    std::map<RPCDCCLink, RPCLBLink> theRPCDCCLinkMap;
    std::pair<RPCAMCLink, RPCLBLink> theRPCAMCLinkPair;
    std::map<RPCAMCLink, RPCLBLink> theRPCAMCLinkMap;
  };
}  // namespace CondFormats_RPCObjects
