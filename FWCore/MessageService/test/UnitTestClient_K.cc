#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_K : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_K(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_K::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    for (int i = 0; i < 10; ++i) {
      edm::LogPrint("cat_P") << "LogPrint: " << i;
      edm::LogSystem("cat_S") << "LogSystem: " << i;
    }
  }

}  // namespace edmtest

using edmtest::UnitTestClient_K;
DEFINE_FWK_MODULE(UnitTestClient_K);
