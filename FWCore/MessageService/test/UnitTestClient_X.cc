#include "FWCore/MessageService/test/UnitTestClient_X.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{

void
  UnitTestClient_X::analyze( edm::Event      const & e
                           , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogWarning("cat_A")   << "LogWarning was used to send this message";
  edm::LogInfo   ("cat_A")   << "LogInfo was used to send this message";
  edm::LogInfo   ("cat_B")   << "LogInfo was used to send this message";
  edm::LogWarning("cat_B")   << "LogWarning was used to send this message";
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_X;
DEFINE_FWK_MODULE(UnitTestClient_X);
