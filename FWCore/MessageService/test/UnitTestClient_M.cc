#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Test of LogSystem, LogAbsolute, LogProblem, LogPrint, LogVerbatim

namespace edmtest {

  class UnitTestClient_M : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_M(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_M::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogSystem("system") << "Text sent to LogSystem";
    edm::LogAbsolute("absolute") << "Text sent to LogAbsolute - should be unformatted";
    edm::LogProblem("problem") << "Text sent to LogProblem - should be unformatted";
    edm::LogPrint("print") << "Text sent to LogPrint- should be unformatted";
    edm::LogVerbatim("verbatim") << "Text sent to LogVerbatim - should be unformatted";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_M;
DEFINE_FWK_MODULE(UnitTestClient_M);
