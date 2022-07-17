#ifndef HeterogeneousCore_SonicTriton_TritonEDFilter
#define HeterogeneousCore_SonicTriton_TritonEDFilter

//TritonDummyCache include comes first for overload resolution
#include "HeterogeneousCore/SonicTriton/interface/TritonDummyCache.h"
#include "HeterogeneousCore/SonicCore/interface/SonicEDFilter.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

//inherited classes that use a non-default GlobalCache should be sure to call the parent initializeGlobalCache()
template <typename G, typename... Capabilities>
class TritonEDFilterT : public SonicEDFilter<TritonClient, edm::GlobalCache<G>, Capabilities...> {
public:
  TritonEDFilterT(edm::ParameterSet const& cfg)
      : SonicEDFilter<TritonClient, edm::GlobalCache<G>, Capabilities...>(cfg) {}

  //use this function to avoid calling TritonService functions Nstreams times
  static std::unique_ptr<G> initializeGlobalCache(edm::ParameterSet const& pset) {
    edm::Service<TritonService> ts;
    const auto& clientPset = pset.getParameterSet("Client");
    ts->addModel(clientPset.getParameter<std::string>("modelName"),
                 clientPset.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
    return nullptr;
  }

  static void globalEndJob(G*) {}

  //destroy client before destructor called to unregister any shared memory before TritonService shuts down fallback server
  virtual void tritonEndStream() {}
  void endStream() final {
    tritonEndStream();
    this->client_.reset();
  }
};

template <typename... Capabilities>
using TritonEDFilter = TritonEDFilterT<TritonDummyCache, Capabilities...>;

#endif
