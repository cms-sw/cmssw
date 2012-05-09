#include "FWCore/MessageService/test/UnitTestClient_Q.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{

void
  UTC_Q1::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogInfo   ("cat_A")   << "Q1 with identifier " << identifier;
  edm::LogInfo   ("timer")   << "Q1 timer with identifier " << identifier;
  edm::LogInfo   ("trace")   << "Q1 trace with identifier " << identifier;
}  

void
  UTC_Q2::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogInfo   ("cat_A")   << "Q2 with identifier " << identifier;
  edm::LogInfo   ("timer")   << "Q2 timer with identifier " << identifier;
  edm::LogInfo   ("trace")   << "Q2 trace with identifier " << identifier;
}  


}  // namespace edmtest

using edmtest::UTC_Q1;
using edmtest::UTC_Q2;
DEFINE_FWK_MODULE(UTC_Q1);
DEFINE_FWK_MODULE(UTC_Q2);
