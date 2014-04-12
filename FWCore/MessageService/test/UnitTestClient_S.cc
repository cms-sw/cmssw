#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageService/test/UnitTestClient_S.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{

bool UTC_S1::enableNotYetCalled = true;
int UTC_S1::n = 0;
int UTC_S2::n = 0;

void
  UTC_S1::analyze( edm::Event      const & /*unused*/
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
  UTC_S2::analyze( edm::Event      const & /*unused*/
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
  UTC_SUMMARY::analyze( edm::Event      const & iEvent
                            , edm::EventSetup const & /*unused*/
                              )
{
  const auto index = iEvent.streamID().value();
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


}  // namespace edmtest

using edmtest::UTC_S1;
using edmtest::UTC_S2;
using edmtest::UTC_SUMMARY;
DEFINE_FWK_MODULE(UTC_S1);
DEFINE_FWK_MODULE(UTC_S2);
DEFINE_FWK_MODULE(UTC_SUMMARY);
