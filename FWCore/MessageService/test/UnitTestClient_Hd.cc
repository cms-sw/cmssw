#ifndef EDM_ML_DEBUG
#define EDM_ML_DEBUG
#endif

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_Hd : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_Hd(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_Hd::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    LogTrace("cat_A") << "LogTrace was used to send this mess"
                      << "age";
    LogDebug("cat_B") << "LogDebug was used to send this other message";
    IfLogTrace(true, "cat_A") << "IfLogTrace was used to send this message";
    IfLogTrace(false, "cat_A") << "IfLogTrace was used to not send this message";
    IfLogDebug(true, "cat_B") << "IfLogDebug was used to send this other message";
    IfLogDebug(false, "cat_B") << "IfLogDebug was used to not send other this message";
    edm::LogVerbatim("cat_A") << "LogVerbatim was us"
                              << "ed to send this message";
    edm::LogInfo("cat_B") << "LogInfo was used to send this other message";
  }

}  // namespace edmtest

using edmtest::UnitTestClient_Hd;
DEFINE_FWK_MODULE(UnitTestClient_Hd);
