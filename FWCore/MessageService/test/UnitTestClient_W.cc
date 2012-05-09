#include "FWCore/MessageService/test/UnitTestClient_W.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{

void
  UnitTestClient_W::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogWarning("cat_A")   << "LogWarning was used to send this message";
  edm::LogInfo   ("cat_B")   << "LogInfo was used to send this message";
  edm::edmmltest::LogWarningThatSuppressesLikeLogInfo("cat_C")
  	<< "LogWarningThatSuppressesLikeLogInfo was used to send this message";
 }  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_W;
DEFINE_FWK_MODULE(UnitTestClient_W);
