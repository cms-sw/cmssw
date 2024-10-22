#include "CondTools/RPC/plugins/RPCInverseTwinMuxLinkMapESProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseTwinMuxLinkMapRcd.h"

RPCInverseTwinMuxLinkMapESProducer::RPCInverseTwinMuxLinkMapESProducer(edm::ParameterSet const& _config) {
  auto cc = setWhatProduced(this);
  es_rpc_tm_l_map_token_ = cc.consumesFrom<RPCAMCLinkMap, RPCTwinMuxLinkMapRcd>();
}

void RPCInverseTwinMuxLinkMapESProducer::fillDescriptions(edm::ConfigurationDescriptions& _descs) {
  edm::ParameterSetDescription _desc;
  _descs.add("RPCInverseTwinMuxLinkMapESProducer", _desc);
}

void RPCInverseTwinMuxLinkMapESProducer::setupRPCTwinMuxLinkMap(RPCTwinMuxLinkMapRcd const& _rcd,
                                                                RPCInverseAMCLinkMap* inverse_linkmap) {
  RPCInverseAMCLinkMap::map_type& _inverse_map(inverse_linkmap->getMap());
  _inverse_map.clear();

  RPCAMCLinkMap const& _map = _rcd.get(es_rpc_tm_l_map_token_);
  for (auto const& _link : _map.getMap()) {
    _inverse_map.insert(RPCInverseAMCLinkMap::map_type::value_type(_link.second, _link.first));
  }
}

std::shared_ptr<RPCInverseAMCLinkMap> RPCInverseTwinMuxLinkMapESProducer::produce(
    RPCInverseTwinMuxLinkMapRcd const& _rcd) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host->ifRecordChanges<RPCTwinMuxLinkMapRcd>(
      _rcd, [this, h = host.get()](auto const& rec) { setupRPCTwinMuxLinkMap(rec, h); });

  return host;
}

//define this as a module
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(RPCInverseTwinMuxLinkMapESProducer);
