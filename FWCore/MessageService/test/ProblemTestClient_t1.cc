#include "FWCore/MessageService/test/ProblemTestClient_t1.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  ProblemTestClient_t1::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
       LogDebug  ("cat_A")   << "This message should not appear";
       LogDebug  ("TrackerGeom")  << "LogDebug was used to send this message";

}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::ProblemTestClient_t1;
DEFINE_FWK_MODULE(ProblemTestClient_t1);
