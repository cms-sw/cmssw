#include "FWCore/MessageService/test/UnitTestClient_B.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{
 
int  UnitTestClient_B::nevent = 0;

void
  UnitTestClient_B::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  nevent++;
  for (int i = 0; i < nevent; ++i) {
    edm::LogError  ("cat_A")   << "LogError was used to send this message";
  }
  edm::LogError  ("cat_B")   << "LogError was used to send this other message";
  edm::LogStatistics();
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_B;
DEFINE_FWK_MODULE(UnitTestClient_B);
