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
  TritonOneEDAnalyzer(edm::ParameterSet const& cfg, const std::string& debugName)
      : SonicOneEDAnalyzer<TritonClient, Capabilities...>(
            cfg, debugName, cfg.getParameterSet("Client").getUntrackedParameter<bool>("verbose")) {
    edm::Service<TritonService> ts;
    const auto& clientPset = cfg.getParameterSet("Client");
    ts->addModel(clientPset.getParameter<std::string>("modelName"),
                 clientPset.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
  }
};

#endif
