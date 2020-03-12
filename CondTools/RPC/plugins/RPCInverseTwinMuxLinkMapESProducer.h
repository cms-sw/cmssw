#ifndef CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h
#define CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
}  // namespace edm

class RPCTwinMuxLinkMapRcd;
class RPCInverseTwinMuxLinkMapRcd;

class RPCInverseTwinMuxLinkMapESProducer : public edm::ESProducer {
public:
  explicit RPCInverseTwinMuxLinkMapESProducer(edm::ParameterSet const& _config);

  static void fillDescriptions(edm::ConfigurationDescriptions& _descs);

  std::shared_ptr<RPCInverseAMCLinkMap> produce(RPCInverseTwinMuxLinkMapRcd const& _rcd);

private:
  using HostType = edm::ESProductHost<RPCInverseAMCLinkMap, RPCTwinMuxLinkMapRcd>;

  void setupRPCTwinMuxLinkMap(RPCTwinMuxLinkMapRcd const&, RPCInverseAMCLinkMap*);

  edm::ReusableObjectHolder<HostType> holder_;
};

#endif  // CondTools_RPC_RPCInverseTwinMuxLinkMapESProducer_h
