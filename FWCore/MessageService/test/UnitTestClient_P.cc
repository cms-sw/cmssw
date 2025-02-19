// Tests effect of LogFlush by cfg-configurable choices of how many 
// messages to use to clog the queue and whether or not FlushMessageLog
// is invoked.  Under normal testing, it will invoke FlushMessageLog in
// a situation where its absence would result in different output.

#include "FWCore/MessageService/test/UnitTestClient_P.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_P::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogWarning ("configuration") << "useLogFlush = " << useLogFlush 
  			       << " queueFillers = " << queueFillers;
  std::string longMessage;
  for (int k=0; k<100; k++) {
    longMessage += "Line in long message\n";
  }
  for (int i=0; i< queueFillers; ++i) {
    edm::LogInfo("cat") <<  "message " << i << "\n" << longMessage;
  }

  edm::LogError ("keyMessage") << "This message is issued just before abort";
  if  (useLogFlush)  edm::FlushMessageLog();
  _exit(0);

 }  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_P;
DEFINE_FWK_MODULE(UnitTestClient_P);
