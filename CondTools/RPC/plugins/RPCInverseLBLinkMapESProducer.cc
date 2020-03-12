#include "CondTools/RPC/plugins/RPCInverseLBLinkMapESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseLBLinkMapRcd.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

RPCInverseLBLinkMapESProducer::RPCInverseLBLinkMapESProducer(edm::ParameterSet const& _config) {
  setWhatProduced(this);
}

void RPCInverseLBLinkMapESProducer::fillDescriptions(edm::ConfigurationDescriptions& _descs) {
  edm::ParameterSetDescription _desc;
  _descs.add("RPCInverseLBLinkMapESProducer", _desc);
}

void RPCInverseLBLinkMapESProducer::setupRPCLBLinkMap(RPCLBLinkMapRcd const& _rcd,
                                                      RPCInverseLBLinkMap* inverse_linkmap) {
  RPCInverseLBLinkMap::map_type& _inverse_map(inverse_linkmap->getMap());
  _inverse_map.clear();

  edm::ESHandle<RPCLBLinkMap> _es_map;
  _rcd.get(_es_map);
  RPCLBLinkMap const& _map = *(_es_map.product());
  for (auto const& _link : _map.getMap()) {
    _inverse_map.insert(RPCInverseLBLinkMap::map_type::value_type(_link.second.getRPCDetId().rawId(), _link));
  }
}

std::shared_ptr<RPCInverseLBLinkMap> RPCInverseLBLinkMapESProducer::produce(RPCInverseLBLinkMapRcd const& _rcd) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host->ifRecordChanges<RPCLBLinkMapRcd>(_rcd, [this, h = host.get()](auto const& rec) { setupRPCLBLinkMap(rec, h); });

  return host;
}

//define this as a module
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(RPCInverseLBLinkMapESProducer);
