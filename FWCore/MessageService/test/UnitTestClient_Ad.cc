#define EDM_ML_DEBUG

#include "FWCore/MessageService/test/UnitTestClient_Ad.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest {

  void UnitTestClient_Ad::analyze(edm::Event const& /*unused*/
                                  ,
                                  edm::EventSetup const& /*unused*/
  ) {
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
    edm::LogInfo("FwkJob") << "<Message>LogInfo was used to send a job report</Message>";

  }  // MessageLoggerClient::analyze()

}  // namespace edmtest

using edmtest::UnitTestClient_Ad;
DEFINE_FWK_MODULE(UnitTestClient_Ad);
