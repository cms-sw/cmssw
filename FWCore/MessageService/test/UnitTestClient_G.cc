#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <iomanip>
#include <iostream>

namespace edmtest {

  class UnitTestClient_G : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_G(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_G::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    if (!edm::isMessageProcessingSetUp()) {
      std::cerr << "??? It appears that Message Processing is not Set Up???\n\n";
    }

    double d = 3.14159265357989;
    edm::LogWarning("cat_A") << "Test of std::setprecision(p):"
                             << " Pi with precision 12 is " << std::setprecision(12) << d;

    for (int i = 0; i < 10; ++i) {
      edm::LogInfo("cat_B") << "\t\tEmit Info level message " << i + 1;
    }

    for (int i = 0; i < 15; ++i) {
      edm::LogWarning("cat_C") << "\t\tEmit Warning level message " << i + 1;
    }
  }

}  // namespace edmtest

using edmtest::UnitTestClient_G;
DEFINE_FWK_MODULE(UnitTestClient_G);
