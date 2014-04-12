#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageService/test/UnitTestClient_SLumi.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{

bool UTC_SL1::enableNotYetCalled = true;
int UTC_SL1::n = 0;
int UTC_SL2::n = 0;

void
  UTC_SL1::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  if (enableNotYetCalled) {
    edm::EnableLoggedErrorsSummary();
    enableNotYetCalled = false;
  }
  n++;
  if (n <= 2) return;
  edm::LogError   ("cat_A")   << "S1 with identifier " << identifier
  			      << " n = " << n;
  edm::LogError   ("grouped_cat")  << "S1 timer with identifier " << identifier;
}  

void
  UTC_SL2::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  n++;
  if (n <= 2) return;
  edm::LogError   ("cat_A")   << "S2 with identifier " << identifier;
  edm::LogError   ("grouped_cat") << "S2 timer with identifier " << identifier;
  edm::LogError   ("cat_B")   << "S2B with identifier " << identifier;
  for (int i = 0; i<n; ++i) {
    edm::LogError   ("cat_B")   << "more S2B";
  }
}  

void
  UTC_SLUMMARY::analyze( edm::Event      const & iEvent
                            , edm::EventSetup const & /*unused*/
                              )
{
  auto const index = iEvent.streamID().value();
  if (!edm::FreshErrorsExist(index)) {
    edm::LogInfo   ("NoFreshErrors") << "Not in this event, anyway";
  }
  std::vector<edm::ErrorSummaryEntry> es = edm::LoggedErrorsSummary(index);
  std::ostringstream os;
  for (unsigned int i = 0; i != es.size(); ++i) {
    os << es[i].category << "   " << es[i].module << "   " 
       << es[i].count << "\n";
  }
  edm::LogVerbatim ("ErrorsInEvent") << os.str();
}  

void
  UTC_SLUMMARY::endLuminosityBlock( edm::LuminosityBlock const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  // throw cms::Exception("endLuminosityBlock called!");
  edm::LogInfo ("endLuminosityBlock") << "endLuminosityBlock() called";
}  


}  // namespace edmtest

using edmtest::UTC_SL1;
using edmtest::UTC_SL2;
using edmtest::UTC_SLUMMARY;
DEFINE_FWK_MODULE(UTC_SL1);
DEFINE_FWK_MODULE(UTC_SL2);
DEFINE_FWK_MODULE(UTC_SLUMMARY);
