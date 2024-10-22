#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edmtest {

  class UnitTestClient_J : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_J(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_J::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::MessageDrop::instance()->debugEnabled = false;

    LogTrace("cat_A") << "LogTrace was used to send this mess"
                      << "age";
    LogDebug("cat_B") << "LogDebug was used to send this other message";
    edm::LogVerbatim("cat_A") << "LogVerbatim was us"
                              << "ed to send this message";
    if (edm::isInfoEnabled())
      edm::LogInfo("cat_B") << "LogInfo was used to send this other message";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_J;
DEFINE_FWK_MODULE(UnitTestClient_J);
