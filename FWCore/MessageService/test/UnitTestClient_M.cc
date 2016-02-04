#include "FWCore/MessageService/test/UnitTestClient_M.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>
#include <iomanip>

// Test of LogSystem, LogAbsolute, LogProblem, LogPrint, LogVerbatim

namespace edmtest
{


void
  UnitTestClient_M::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogSystem("system")    << 
  	"Text sent to LogSystem"; 
  edm::LogAbsolute("absolute")  << 
  	"Text sent to LogAbsolute - should be unformatted"; 
  edm::LogProblem("problem")   << 
  	"Text sent to LogProblem - should be unformatted"; 
  edm::LogPrint("print")       << 
  	"Text sent to LogPrint- should be unformatted"; 
  edm::LogVerbatim("verbatim") << 
  	"Text sent to LogVerbatim - should be unformatted"; 
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_M;
DEFINE_FWK_MODULE(UnitTestClient_M);
