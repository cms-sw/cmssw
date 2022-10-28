#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <string>

namespace edmtest {

  class UnitTestClient_A : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_A(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_A::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    std::string empty_;
    std::string file_ = "nameOfFile";
    LogDebug("cat_A") << "LogDebug was used to send this message";
    LogDebug("cat_B") << "LogDebug was used to send this other message";
    edm::LogError("cat_A") << "LogError was used to send this message"
                           << "-which is long enough to span lines but-"
                           << "will not be broken up by the logger any more";
    edm::LogError("cat_B") << "LogError was used to send this other message";
    edm::LogWarning("cat_A") << "LogWarning was used to send this message";
    edm::LogWarning("cat_B") << "LogWarning was used to send this other message";
    edm::LogInfo("cat_A") << "LogInfo was used to send this message";
    edm::LogInfo("cat_B") << "LogInfo was used to send this other message";
    edm::LogInfo("FwkTest") << "<Message>LogInfo was used to send a job report</Message>";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_A;
DEFINE_FWK_MODULE(UnitTestClient_A);
