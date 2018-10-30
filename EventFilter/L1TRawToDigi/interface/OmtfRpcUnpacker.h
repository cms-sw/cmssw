#ifndef EventFilter_L1TRawToDigi_Omtf_RpcUnpacker_H
#define EventFilter_L1TRawToDigi_Omtf_RpcUnpacker_H

#include <string>
#include <memory>

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/L1TMuon/interface/OMTF/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingRpc.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h" 

namespace edm { class EventSetup; }
namespace omtf { class RpcDataWord64; }

namespace omtf {

class RpcUnpacker {

public:
  RpcUnpacker() {}

  void init(const edm::EventSetup & es);
  void init(const edm::EventSetup & es, const std::string & connectionFile);
  void unpack(int triggerBX, unsigned int fed, unsigned int amc, const RpcDataWord64 &raw, RPCDigiCollection * prod);

private:
  void initCabling(const edm::EventSetup & es);

  MapEleIndex2LBIndex      theOmtf2Pact;
  std::unique_ptr<const RPCReadOutMapping> thePactCabling;
};
}
#endif

