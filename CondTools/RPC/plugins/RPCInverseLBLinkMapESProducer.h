#ifndef CondTools_RPC_RPCInverseLBLinkMapESProducer_h
#define CondTools_RPC_RPCInverseLBLinkMapESProducer_h

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include "CondFormats/RPCObjects/interface/RPCInverseLBLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
}  // namespace edm

class RPCLBLinkMapRcd;
class RPCInverseLBLinkMapRcd;

class RPCInverseLBLinkMapESProducer : public edm::ESProducer {
public:
  explicit RPCInverseLBLinkMapESProducer(edm::ParameterSet const& _config);

  static void fillDescriptions(edm::ConfigurationDescriptions& _descs);

  std::shared_ptr<RPCInverseLBLinkMap> produce(RPCInverseLBLinkMapRcd const& _rcd);

private:
  using HostType = edm::ESProductHost<RPCInverseLBLinkMap, RPCLBLinkMapRcd>;

  void setupRPCLBLinkMap(RPCLBLinkMapRcd const&, RPCInverseLBLinkMap*);

  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<RPCLBLinkMap, RPCLBLinkMapRcd> es_rpc_lb_map_token_;
};

#endif  // CondTools_RPC_RPCInverseLBLinkMapESProducer_h
