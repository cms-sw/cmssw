#include "FWCore/MessageService/test/UnitTestClient_H.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_H::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
#ifdef EDM_ML_DEBUG
       const bool debug1 = true;
       const bool debug2 = false;
#endif

       LogTrace    ("cat_A") << "LogTrace was used to send this mess" << "age";
       LogDebug    ("cat_B") << "LogDebug was used to send this other message";
       IfLogTrace(debug1, "cat_A") << "IfLogTrace was used to send this message";
       IfLogTrace(debug2, "cat_A") << "IfLogTrace was used to not send this message";
       IfLogDebug(debug1, "cat_B") << "IfLogDebug was used to send this other message";
       IfLogDebug(debug2, "cat_B") << "IfLogDebug was used to not send other this message";
  edm::LogVerbatim ("cat_A") << "LogVerbatim was us" << "ed to send this message";
  edm::LogInfo     ("cat_B") << "LogInfo was used to send this other message";
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_H;
DEFINE_FWK_MODULE(UnitTestClient_H);
