#include "FWCore/MessageService/test/UnitTestClient_L.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <iostream>
#include <string>


namespace edmtest
{


void UnitTestClient_L::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  for (int i=0; i<10000000; ++i) {
  }
  edm::LogInfo     ("cat") << "Event " << e.id() << "complete";
}  // MessageLoggerClient::analyze()

void UnitTestClient_L1::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  for (int i=0; i<10000000; ++i) {
       LogDebug    ("dog") << "I am perhaps creating a long string here";
  }
  edm::LogInfo     ("cat") << "Event " << e.id() << "complete";
}  // MessageLoggerClient::analyze()

}  // namespace edmtest


using edmtest::UnitTestClient_L;
using edmtest::UnitTestClient_L1;
DEFINE_FWK_MODULE(UnitTestClient_L);
DEFINE_FWK_MODULE(UnitTestClient_L1);
