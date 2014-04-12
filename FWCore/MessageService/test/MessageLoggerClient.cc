#include "FWCore/MessageService/test/MessageLoggerClient.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>


namespace edmtest
{


void
  MessageLoggerClient::analyze( edm::Event      const & /*unused*/
                              , edm::EventSetup const & /*unused*/
                              )
{
  //std::cout << "Module reached\n";
  LogDebug("aTestMessage") << "LogDebug was used to send this message";
  edm::LogInfo("aTestMessage") << "LogInfo was used to send this message";
  edm::LogWarning("aTestMessage") << "LogWarning was used to send this message";
  edm::LogError("aTestMessage") << "LogError was used to send this message";
  edm::LogInfo("cat1|cat2||cat3") << "Three-category message";

  edm::LogWarning("aboutToSend") << "about to send 100 warnings";
  for( unsigned i = 0;  i != 100;  ++i )  {
    edm::LogWarning("unimportant") << "warning number " << i;
  }


}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::MessageLoggerClient;
DEFINE_FWK_MODULE(MessageLoggerClient);
