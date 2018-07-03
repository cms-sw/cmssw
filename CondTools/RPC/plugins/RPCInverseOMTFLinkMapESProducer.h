#ifndef CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h
#define CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCOMTFLinkMapRcd;
class RPCInverseOMTFLinkMapRcd;

class RPCInverseOMTFLinkMapESProducer
    : public edm::ESProducer
{
public:
    explicit RPCInverseOMTFLinkMapESProducer(edm::ParameterSet const & _config);

    static void fillDescriptions(edm::ConfigurationDescriptions & _descs);

    void RPCOMTFLinkMapCallback(RPCOMTFLinkMapRcd const & _rcd);

    std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseOMTFLinkMapRcd const & _rcd);

protected:
    std::shared_ptr<RPCInverseAMCLinkMap> inverse_linkmap_;
};

#endif // CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h
