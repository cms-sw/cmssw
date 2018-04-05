#ifndef CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h
#define CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCCPPFLinkMapRcd;
class RPCInverseCPPFLinkMapRcd;

class RPCInverseCPPFLinkMapESProducer
    : public edm::ESProducer
{
public:
    explicit RPCInverseCPPFLinkMapESProducer(edm::ParameterSet const & _config);

    static void fillDescriptions(edm::ConfigurationDescriptions & _descs);

    void RPCCPPFLinkMapCallback(RPCCPPFLinkMapRcd const & _rcd);

    std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseCPPFLinkMapRcd const & _rcd);

protected:
    std::shared_ptr<RPCInverseAMCLinkMap> inverse_linkmap_;
};

#endif // CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h
