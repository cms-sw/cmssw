#ifndef CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h
#define CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCTwinMuxLinkMapRcd;
class RPCInverseTwinMuxLinkMapRcd;

class RPCInverseTwinMuxLinkMapESProducer
    : public edm::ESProducer
{
public:
    explicit RPCInverseTwinMuxLinkMapESProducer(edm::ParameterSet const & _config);

    static void fillDescriptions(edm::ConfigurationDescriptions & _descs);

    void RPCTwinMuxLinkMapCallback(RPCTwinMuxLinkMapRcd const & _rcd);

    std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseTwinMuxLinkMapRcd const & _rcd);

protected:
    std::shared_ptr<RPCInverseAMCLinkMap> inverse_linkmap_;
};

#endif // CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h
