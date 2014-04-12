#include "FWCore/MessageService/test/UnitTestClient_J.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_J::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{

  edm::MessageDrop::instance()->debugEnabled  = false;

       LogTrace    ("cat_A") << "LogTrace was used to send this mess" << "age";
       LogDebug    ("cat_B") << "LogDebug was used to send this other message";
  edm::LogVerbatim ("cat_A") << "LogVerbatim was us" << "ed to send this message";
  if( edm::isInfoEnabled() ) 
     edm::LogInfo  ("cat_B") << "LogInfo was used to send this other message";
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_J;
DEFINE_FWK_MODULE(UnitTestClient_J);
