#include "CondTools/RPC/plugins/RPCInverseLBLinkMapESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseLBLinkMapRcd.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

RPCInverseLBLinkMapESProducer::RPCInverseLBLinkMapESProducer(edm::ParameterSet const & _config)
    : inverse_linkmap_(new RPCInverseLBLinkMap())
{
    setWhatProduced(this, edm::eventsetup::dependsOn(&RPCInverseLBLinkMapESProducer::RPCLBLinkMapCallback));
}

void RPCInverseLBLinkMapESProducer::fillDescriptions(edm::ConfigurationDescriptions & _descs)
{
    edm::ParameterSetDescription _desc;
    _descs.add("RPCInverseLBLinkMapESProducer", _desc);
}

void RPCInverseLBLinkMapESProducer::RPCLBLinkMapCallback(RPCLBLinkMapRcd const & _rcd)
{
    RPCInverseLBLinkMap::map_type & _inverse_map(inverse_linkmap_->getMap());
    _inverse_map.clear();

    edm::ESHandle<RPCLBLinkMap> _es_map;
    _rcd.get(_es_map);
    RPCLBLinkMap const & _map = *(_es_map.product());
    for (auto const & _link : _map.getMap()) {
        _inverse_map.insert(RPCInverseLBLinkMap::map_type::value_type(_link.second.getRPCDetId().rawId(), _link));
    }
}

std::shared_ptr<RPCInverseLBLinkMap> RPCInverseLBLinkMapESProducer::produce(RPCInverseLBLinkMapRcd const & _rcd)
{
    return inverse_linkmap_;
}

//define this as a module
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(RPCInverseLBLinkMapESProducer);
