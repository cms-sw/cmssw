#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_R : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_R(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_R::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    for (int i = 0; i < 10000; ++i) {
      edm::LogError("cat_A") << "A " << i;
      edm::LogError("cat_B") << "B " << i;
    }
  }

}  // namespace edmtest

using edmtest::UnitTestClient_R;
DEFINE_FWK_MODULE(UnitTestClient_R);
