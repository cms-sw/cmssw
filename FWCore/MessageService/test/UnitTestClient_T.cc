#include "FWCore/MessageService/test/UnitTestClient_T.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>
#include <sstream>

namespace edmtest
{

void
  UTC_T1::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  if (ev == 0) edm::EnableLoggedErrorsSummary();
  edm::LogError  ("cat_A")   << "T1 error with identifier " << identifier
  			     << " event " << ev;
  edm::LogWarning("cat_A")   << "T1 warning with identifier " << identifier
  			     << " event " << ev;
  edm::LogError  ("timer")   << "T1 timer error with identifier " << identifier
  			     << " event " << ev;
  ev++;
}  

void
  UTC_T2::analyze( edm::Event      const & /*unused*/
                            , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogError  ("cat_A")   << "T2 error with identifier " << identifier
  			     << " event " << ev;
  edm::LogWarning("cat_A")   << "T2 warning with identifier " << identifier
  			     << " event " << ev;
  edm::LogError  ("timer")   << "T2 timer error with identifier " << identifier
  			     << " event " << ev;
  if ( ev == 9 ) {
    if (edm::FreshErrorsExist()) {
	edm::LogInfo("summary") << 
	"At ev = " << ev << "FreshErrorsExist() returns true"; 
    } else {
	edm::LogError("summary") 
	<< "At ev = " << ev << "FreshErrorsExist() returns false"
	<< " which is unexpected"; 
    }
    std::vector<edm::ErrorSummaryEntry> v = edm::LoggedErrorsSummary();
    if (!edm::FreshErrorsExist()) {
	edm::LogInfo("summary") << 
		"After LoggedErrorsSummary() FreshErrorsExist() returns false"; 
    } else {
	edm::LogError("summary") << 
	    "After LoggedErrorsSummary() FreshErrorsExist() returns true"
		     << " which is unexpected"; 
    }
    printLES(v);
  }
  if ( ev == 15 ) {
    if (edm::FreshErrorsExist()) {
	edm::LogInfo("summary") 
	<< "At ev = " << ev << "FreshErrorsExist() returns true"; 
    } else {
	edm::LogError("summary") 
	<< "At ev = " << ev << "FreshErrorsExist() returns false"
	<< " which is unexpected"; 
    }
    std::vector<edm::ErrorSummaryEntry> v = edm::LoggedErrorsOnlySummary();
    if (!edm::FreshErrorsExist()) {
	edm::LogInfo("summary") << 
	"After LoggedErrorsOnlySummary() FreshErrorsExist() returns false"; 
    } else {
	edm::LogError("summary") << 
	    "After LoggedErrorsOnlySummary() FreshErrorsExist() returns true"
		     << " which is unexpected"; 
    }
    printLES(v);
  }
  ev++;
}  

void
UTC_T2::printLES(std::vector<edm::ErrorSummaryEntry> const & v) {
  std::ostringstream s;
  typedef std::vector<edm::ErrorSummaryEntry>::const_iterator IT;
  IT end = v.end();
  s << "Error Summary Vector with " << v.size() << " entries:\n";
  for (IT i = v.begin(); i != end; ++i) {
    s << "Category " << i->category << "   Module " << i->module
    << "   Severity " << (i->severity).getName() 
    << "   Count " << i->count << "\n";
  }   
  s << "-------------------------- \n";
  edm::LogVerbatim("summary") << s.str();
}

}  // namespace edmtest

using edmtest::UTC_T1;
using edmtest::UTC_T2;
DEFINE_FWK_MODULE(UTC_T1);
DEFINE_FWK_MODULE(UTC_T2);
