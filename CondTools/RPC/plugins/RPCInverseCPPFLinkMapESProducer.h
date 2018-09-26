#ifndef CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h
#define CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

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

    std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseCPPFLinkMapRcd const & _rcd);

private:

    using HostType = edm::ESProductHost<RPCInverseAMCLinkMap,
                                        RPCCPPFLinkMapRcd>;

    void setupRPCCPPFLinkMap(RPCCPPFLinkMapRcd const&,
                             RPCInverseAMCLinkMap*);

    edm::ReusableObjectHolder<HostType> holder_;
};

#endif // CondTools_RPC_RPCInverseCPPFLinkMapESProducer_h
