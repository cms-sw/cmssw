#ifndef CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h
#define CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

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

    std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseOMTFLinkMapRcd const & _rcd);

private:

    using HostType = edm::ESProductHost<RPCInverseAMCLinkMap,
                                        RPCOMTFLinkMapRcd>;

    void setupRPCOMTFLinkMap(RPCOMTFLinkMapRcd const&,
                             RPCInverseAMCLinkMap*);

    edm::ReusableObjectHolder<HostType> holder_;
};

#endif // CondTools_RPC_RPCInverseOMTFLinkMapESProducer_h
