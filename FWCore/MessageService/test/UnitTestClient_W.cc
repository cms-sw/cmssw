#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {
  class UnitTestClient_W : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_W(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_W::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogWarning("cat_A") << "LogWarning was used to send this message";
    edm::LogInfo("cat_B") << "LogInfo was used to send this message";
    edm::edmmltest::LogWarningThatSuppressesLikeLogInfo("cat_C")
        << "LogWarningThatSuppressesLikeLogInfo was used to send this message";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_W;
DEFINE_FWK_MODULE(UnitTestClient_W);
