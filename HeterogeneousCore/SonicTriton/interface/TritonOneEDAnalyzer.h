#ifndef HeterogeneousCore_SonicTriton_TritonOneEDAnalyzer
#define HeterogeneousCore_SonicTriton_TritonOneEDAnalyzer

#include "HeterogeneousCore/SonicCore/interface/SonicOneEDAnalyzer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

#include <string>

template <typename... Capabilities>
class TritonOneEDAnalyzer : public SonicOneEDAnalyzer<TritonClient, Capabilities...> {
public:
  TritonOneEDAnalyzer(edm::ParameterSet const& cfg) : SonicOneEDAnalyzer<TritonClient, Capabilities...>(cfg) {
    edm::Service<TritonService> ts;
    ts->addModel(this->clientPset_.template getParameter<std::string>("modelName"),
                 this->clientPset_.template getParameter<edm::FileInPath>("modelConfigPath").fullPath());
  }

  //destroy client before destructor called to unregister any shared memory before TritonService shuts down fallback server
  virtual void tritonEndJob() {}
  void endJob() final {
    tritonEndJob();
    this->client_.reset();
  }
};

#endif
