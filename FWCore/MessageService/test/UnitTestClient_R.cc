#include "FWCore/MessageService/test/UnitTestClient_R.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>
#include <iomanip>

namespace edmtest
{


void
  UnitTestClient_R::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{

  for( int i=0; i<10000; ++i) {
    edm::LogError("cat_A")   << "A " << i;
    edm::LogError("cat_B")   << "B " << i;
  }
}  // MessageLoggerClient::analyze()

}  // namespace edmtest


using edmtest::UnitTestClient_R;
DEFINE_FWK_MODULE(UnitTestClient_R);
