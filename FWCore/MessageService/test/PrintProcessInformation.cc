#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

namespace edmtest {
  class PrintProcessInformation : public edm::global::EDAnalyzer<> {
  public:
    explicit PrintProcessInformation(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const final {
      auto const* processConfiguration =
          e.moduleCallingContext()->getStreamContext()->processContext()->processConfiguration();
      assert(processConfiguration);
      auto reduced = *processConfiguration;
      reduced.reduce();
      edm::LogSystem("PrintProcessInformation")
          << "Name:" << processConfiguration->processName() << "\nReducedConfigurationID:" << reduced.id()
          << "\nParameterSetID:" << processConfiguration->parameterSetID();
    }
  };
}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::PrintProcessInformation);
