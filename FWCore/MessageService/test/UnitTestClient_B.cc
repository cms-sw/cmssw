// UnitTestClient_B is used for testing LogStatistics and the reset behaviors
// of statistics destinations.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edmtest {

  class UnitTestClient_B : public edm::one::EDAnalyzer<> {
  public:
    explicit UnitTestClient_B(edm::ParameterSet const&) {}

    void analyze(edm::Event const&, edm::EventSetup const&) override;

  private:
    int nevent = 0;
  };

  void UnitTestClient_B::analyze(edm::Event const&, edm::EventSetup const&) {
    nevent++;
    for (int i = 0; i < nevent; ++i) {
      edm::LogError("cat_A") << "LogError was used to send this message";
    }
    edm::LogError("cat_B") << "LogError was used to send this other message";
    edm::LogStatistics();
  }

}  // namespace edmtest

using edmtest::UnitTestClient_B;
DEFINE_FWK_MODULE(UnitTestClient_B);
