#include "CondTools/RPC/plugins/RPCInverseTwinMuxLinkMapESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseTwinMuxLinkMapRcd.h"

RPCInverseTwinMuxLinkMapESProducer::RPCInverseTwinMuxLinkMapESProducer(edm::ParameterSet const & _config)
{
    setWhatProduced(this);
}

void RPCInverseTwinMuxLinkMapESProducer::fillDescriptions(edm::ConfigurationDescriptions & _descs)
{
    edm::ParameterSetDescription _desc;
    _descs.add("RPCInverseTwinMuxLinkMapESProducer", _desc);
}

void RPCInverseTwinMuxLinkMapESProducer::setupRPCTwinMuxLinkMap(RPCTwinMuxLinkMapRcd const & _rcd,
                                                                RPCInverseAMCLinkMap* inverse_linkmap)
{
    RPCInverseAMCLinkMap::map_type & _inverse_map(inverse_linkmap->getMap());
    _inverse_map.clear();

    edm::ESHandle<RPCAMCLinkMap> _es_map;
    _rcd.get(_es_map);
    RPCAMCLinkMap const & _map = *(_es_map.product());
    for (auto const & _link : _map.getMap()) {
        _inverse_map.insert(RPCInverseAMCLinkMap::map_type::value_type(_link.second, _link.first));
    }
}

std::shared_ptr<RPCInverseAMCLinkMap> RPCInverseTwinMuxLinkMapESProducer::produce(RPCInverseTwinMuxLinkMapRcd const & _rcd)
{
    auto host = holder_.makeOrGet([]() {
        return new HostType;
    });

    host->ifRecordChanges<RPCTwinMuxLinkMapRcd>(_rcd,
                                                [this,h=host.get()](auto const& rec) {
        setupRPCTwinMuxLinkMap(rec, h);
    });

    return host;
}

//define this as a module
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(RPCInverseTwinMuxLinkMapESProducer);
