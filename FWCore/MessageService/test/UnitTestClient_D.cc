#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_D : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_D(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_D::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogWarning("cat_A") << "This message should not appear in "
                             << "the framework job report";
    edm::LogWarning("FwkTest") << "<Message>This message should appear in "
                               << "the framework job report</Message>";
    edm::LogWarning("special") << "This message should appear in "
                               << "restrict but the others should not";
  }
}  // namespace edmtest

using edmtest::UnitTestClient_D;
DEFINE_FWK_MODULE(UnitTestClient_D);
