#ifndef EventFilter_RPCRawToDigi_RPCCPPFUnpacker_h
#define EventFilter_RPCRawToDigi_RPCCPPFUnpacker_h

#include <map>
#include <set>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "CondFormats/DataRecord/interface/RPCCPPFLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMC13Record.h"
#include "EventFilter/RPCRawToDigi/interface/RPCCPPFRecord.h"
#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpacker.h"

class RPCAMCLinkCounters;

class RPCCPPFUnpacker : public RPCAMCUnpacker {
public:
  RPCCPPFUnpacker(edm::ParameterSet const&, edm::ProducesCollector);

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;
  void produce(edm::Event& event,
               edm::EventSetup const& setup,
               std::map<RPCAMCLink, rpcamc13::AMCPayload> const& amc_payload) override;

protected:
  bool processCPPF(RPCAMCLink const& link,
                   rpcamc13::AMCPayload const& payload,
                   RPCAMCLinkCounters& counters,
                   std::set<std::pair<RPCDetId, RPCDigi> >& rpc_digis,
                   l1t::CPPFDigiCollection& rpc_cppf_digis) const;
  void processRXRecord(RPCAMCLink link,
                       unsigned int bx_counter_mod,
                       rpccppf::RXRecord const& record,
                       RPCAMCLinkCounters& counters,
                       std::set<std::pair<RPCDetId, RPCDigi> >& rpc_digis,
                       int bx_min,
                       int bx_max) const;
  void processTXRecord(RPCAMCLink link,
                       unsigned int block,
                       unsigned int word,
                       rpccppf::TXRecord const& record,
                       l1t::CPPFDigiCollection& rpc_cppf_digis) const;
  void putRPCDigis(edm::Event& event, std::set<std::pair<RPCDetId, RPCDigi> > const& digis) const;

protected:
  bool fill_counters_;
  int bx_min_, bx_max_;

  edm::ESWatcher<RPCCPPFLinkMapRcd> es_cppf_link_map_watcher_;
  edm::ESHandle<RPCAMCLinkMap> es_cppf_link_map_;
  edm::ESHandle<RPCLBLinkMap> es_lb_link_map_;
};

#endif  // EventFilter_RPCRawToDigi_RPCCPPFUnpacker_h
