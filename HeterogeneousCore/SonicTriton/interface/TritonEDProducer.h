#ifndef HeterogeneousCore_SonicTriton_TritonEDProducer
#define HeterogeneousCore_SonicTriton_TritonEDProducer

//TritonDummyCache include comes first for overload resolution
#include "HeterogeneousCore/SonicTriton/interface/TritonDummyCache.h"
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

//inherited classes that use a non-default GlobalCache should be sure to call the parent initializeGlobalCache()
template <typename G, typename... Capabilities>
class TritonEDProducerT : public SonicEDProducer<TritonClient, edm::GlobalCache<G>, Capabilities...> {
public:
  TritonEDProducerT(edm::ParameterSet const& cfg)
      : SonicEDProducer<TritonClient, edm::GlobalCache<G>, Capabilities...>(cfg) {}

  //use this function to avoid calling TritonService functions Nstreams times
  static std::unique_ptr<G> initializeGlobalCache(edm::ParameterSet const& pset) {
    edm::Service<TritonService> ts;
    const auto& clientPset = pset.getParameterSet("Client");
    ts->addModel(clientPset.getParameter<std::string>("modelName"),
                 clientPset.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
    return nullptr;
  }

  static void globalEndJob(G*) {}
};

template <typename... Capabilities>
using TritonEDProducer = TritonEDProducerT<TritonDummyCache, Capabilities...>;

#endif
