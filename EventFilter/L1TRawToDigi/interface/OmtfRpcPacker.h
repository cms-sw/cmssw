#ifndef EventFilter_L1TRawToDigi_Omtf_RpcPacker_H
#define EventFilter_L1TRawToDigi_Omtf_RpcPacker_H

#include <string>

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingRpc.h"

namespace edm { class EventSetup; }

namespace omtf {

class RpcPacker {

public:
  RpcPacker() : thePactCabling(0) {}

  void init(const edm::EventSetup & es);
  void init(const edm::EventSetup & es, const std::string & connectionFile);
  void pack(const RPCDigiCollection * prod, FedAmcRawsMap & raws);

private:
  void initCabling(const edm::EventSetup & es);

  MapLBIndex2EleIndex      thePact2Omtf;
  const RPCReadOutMapping* thePactCabling;
};
}
#endif
