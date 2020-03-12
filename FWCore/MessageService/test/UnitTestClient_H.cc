#ifdef EDM_ML_DEBUG
#undef EDM_ML_DEBUG
#endif

#include "FWCore/MessageService/test/UnitTestClient_H.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest {

  void UnitTestClient_H::analyze(edm::Event const& /*unused*/
                                 ,
                                 edm::EventSetup const& /*unused*/
  ) {
    LogTrace("cat_A") << "LogTrace was used to send this mess"
                      << "age";
    LogDebug("cat_B") << "LogDebug was used to send this other message";
    IfLogTrace(nonexistent, "cat_A")
        << "IfLogTrace was used to send this message";  // check that these compile with non-existent variable as the condition
    IfLogTrace(nonexistent, "cat_A") << "IfLogTrace was used to not send this message";
    IfLogDebug(nonexistent, "cat_B") << "IfLogDebug was used to send this other message";
    IfLogDebug(nonexistent, "cat_B") << "IfLogDebug was used to not send other this message";
    edm::LogVerbatim("cat_A") << "LogVerbatim was us"
                              << "ed to send this message";
    edm::LogInfo("cat_B") << "LogInfo was used to send this other message";
  }  // MessageLoggerClient::analyze()

}  // namespace edmtest

using edmtest::UnitTestClient_H;
DEFINE_FWK_MODULE(UnitTestClient_H);
