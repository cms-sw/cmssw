#ifndef CondTools_RPC_RPCInverseLBLinkMapESProducer_h
#define CondTools_RPC_RPCInverseLBLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RPCObjects/interface/RPCInverseLBLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCLBLinkMapRcd;
class RPCInverseLBLinkMapRcd;

class RPCInverseLBLinkMapESProducer
    : public edm::ESProducer
{
public:
    explicit RPCInverseLBLinkMapESProducer(edm::ParameterSet const & _config);

    static void fillDescriptions(edm::ConfigurationDescriptions & _descs);

    void RPCLBLinkMapCallback(RPCLBLinkMapRcd const & _rcd);

    std::shared_ptr<RPCInverseLBLinkMap> produce(RPCInverseLBLinkMapRcd const & _rcd);

protected:
    std::shared_ptr<RPCInverseLBLinkMap> inverse_linkmap_;
};

#endif // CondTools_RPC_RPCInverseLBLinkMapESProducer_h
