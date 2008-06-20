#include "FWCore/MessageService/test/UnitTestClient_G.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>
#include <iomanip>

namespace edmtest
{


void
  UnitTestClient_G::analyze( edm::Event      const & e
                           , edm::EventSetup const & /*unused*/
                              )
{
  if (!edm::isMessageProcessingSetUp()) {
    std::cerr << "??? It appears that Message Processing is not Set Up???\n\n";
  }

  double d = 3.14159265357989;
  edm::LogWarning("cat_A")   << "Test of std::setprecision(p):"
  			     << " Pi with precision 12 is " 
  			     << std::setprecision(12) << d;

  for( int i=0; i<10; ++i) {
    edm::LogInfo("cat_B")      << "\t\tEmit Info level message " << i+1;
  }

  for( int i=0; i<15; ++i) {
    edm::LogWarning("cat_C")      << "\t\tEmit Warning level message " << i+1;
  }
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_G;
DEFINE_FWK_MODULE(UnitTestClient_G);
