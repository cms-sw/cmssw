#include "CondTools/RPC/plugins/RPCInverseOMTFLinkMapESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseOMTFLinkMapRcd.h"

RPCInverseOMTFLinkMapESProducer::RPCInverseOMTFLinkMapESProducer(edm::ParameterSet const & _config)
    : inverse_linkmap_(new RPCInverseAMCLinkMap())
{
    setWhatProduced(this, edm::eventsetup::dependsOn(&RPCInverseOMTFLinkMapESProducer::RPCOMTFLinkMapCallback));
}

void RPCInverseOMTFLinkMapESProducer::fillDescriptions(edm::ConfigurationDescriptions & _descs)
{
    edm::ParameterSetDescription _desc;
    _descs.add("RPCInverseOMTFLinkMapESProducer", _desc);
}

void RPCInverseOMTFLinkMapESProducer::RPCOMTFLinkMapCallback(RPCOMTFLinkMapRcd const & _rcd)
{
    RPCInverseAMCLinkMap::map_type & _inverse_map(inverse_linkmap_->getMap());
    _inverse_map.clear();

    edm::ESHandle<RPCAMCLinkMap> _es_map;
    _rcd.get(_es_map);

    RPCAMCLinkMap const & _map = *(_es_map.product());
    for (auto const & _link : _map.getMap()) {
        _inverse_map.insert(RPCInverseAMCLinkMap::map_type::value_type(_link.second, _link.first));
    }
}

std::shared_ptr<RPCInverseAMCLinkMap> RPCInverseOMTFLinkMapESProducer::produce(RPCInverseOMTFLinkMapRcd const & _rcd)
{
    return inverse_linkmap_;
}

//define this as a module
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(RPCInverseOMTFLinkMapESProducer);
