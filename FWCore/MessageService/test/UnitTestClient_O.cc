#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_O : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_O(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_O::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogInfo("importantInfo") << "This LogInfo message should appear in both destinations";
    edm::LogInfo("routineInfo") << "This LogInfo message should appear in the info destination";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_O;
DEFINE_FWK_MODULE(UnitTestClient_O);
