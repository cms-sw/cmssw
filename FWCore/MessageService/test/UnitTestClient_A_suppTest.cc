// UnitTestClient_A_suppTest is a variant on UnitTestClient_A
// that adds a function call (foo()) to the end of the statemnts 
// of a LogDebug and a LogInfo.  foo() has a side efffect 
// (a write to cerr).  What we are testing is whether when LogDebug
// is suppressed because EDM_ML_DEBUG is not defined, foo() is not
// called. 
// 
// The correct behavior is that cerr will get two lines of 
// foo(LogInfo) was called.
// But cerr should get no lines of 
// foo(LogDebug) was called.
// 
// Test passed 9/27/10

#include "FWCore/MessageService/test/UnitTestClient_A.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

// TEMPORARY - to test suppression
static std::string foo(std::string const & x) {
  std::cerr << "foo(" << x << ") was called.\n";
  return std::string();
}

namespace edmtest
{

void
  UnitTestClient_A::analyze( edm::Event      const & e
                           , edm::EventSetup const & /*unused*/
                              )
{
  std::string empty_;
  std::string file_ = "nameOfFile";
       LogDebug  ("cat_A")   << "LogDebug was used to send this message"
// TEMPORARY change to test suppression
	<< foo("LogDebug");
       LogDebug  ("cat_B")   << "LogDebug was used to send this other message";
  edm::LogError  ("cat_A")   << "LogError was used to send this message"
  			     << "-which is long enough to span lines but-"
			     << "will not be broken up by the logger any more";
  edm::LogError  ("cat_B")   << "LogError was used to send this other message";
  edm::LogWarning("cat_A")   << "LogWarning was used to send this message";
  edm::LogWarning("cat_B")   << "LogWarning was used to send this other message";
  edm::LogInfo   ("cat_A")   << "LogInfo was used to send this message"
// TEMPORARY change to test suppression
      << foo("LogInfo");
  edm::LogInfo   ("cat_B")   << "LogInfo was used to send this other message";
  edm::LogInfo   ("FwkJob")  << "<Message>LogInfo was used to send a job report</Message>";

 }  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_A;
DEFINE_FWK_MODULE(UnitTestClient_A);
