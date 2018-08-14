#ifndef EventFilter_L1TRawToDigi_Omtf_LinkMappingRpc_H
#define EventFilter_L1TRawToDigi_Omtf_LinkMappingRpc_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "EventFilter/L1TRawToDigi/interface/OmtfEleIndex.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"

class RPCReadOutMapping;
namespace edm { class EventSetup; }

namespace omtf {

  class RpcLinkMap;


  typedef std::map<EleIndex, LinkBoardElectronicIndex> MapEleIndex2LBIndex;
  MapEleIndex2LBIndex translateOmtf2Pact(const RpcLinkMap & omtfLnks, const RPCReadOutMapping* pactCabling); 

  struct lessLinkBoardElectronicIndex { bool operator() (const LinkBoardElectronicIndex & o1, const LinkBoardElectronicIndex & o2) const; };
  typedef std::map<LinkBoardElectronicIndex, std::pair<EleIndex,EleIndex>, lessLinkBoardElectronicIndex > MapLBIndex2EleIndex;
  MapLBIndex2EleIndex  translatePact2Omtf(const RpcLinkMap & omtfLnks, const RPCReadOutMapping* pactCabling); 

class RpcLinkMap {
public:
  RpcLinkMap() { }

  void init(const edm::EventSetup& es);

  void init( const std::string& fName);

  const std::string & lbName(const std::string& board, unsigned int link) const { return link2lbName.at(board).at(link); }

  unsigned int link(const std::string& board, const std::string& lbName) const { return lbName2link.at(board).at(lbName); }

  std::vector<EleIndex> omtfEleIndex ( const std::string& lbName) const;

private:
    std::map<std::string, std::map<unsigned int, std::string> > link2lbName; //[processorName][rpcRxNum] - lbName
    std::map<std::string, std::map<std::string, unsigned int> > lbName2link; //[processorName][lbName] - rpcRxNum
    std::map<std::string, std::vector<EleIndex> > lbName2OmtfIndex; //[lbName] - vector of {board,rpcRxNum}

};

} // namespace omtf
#endif
