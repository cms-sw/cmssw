#ifndef MessageLogger_LoggedErrorsSummaryy_h
#define MessageLogger_LoggedErrorsSummaryEntry_h

// ----------------------------------------------------------------------
//
// LoggedErrorsSummary.h - Methods to obtain summary of a warning or error
//		    message issued in an event.  
//
//   Usage:  
//	EnableLoggedErrorsSummary();
//	then per event:
//	some_code_that_might_issue_messages();
//	if (edm::FreshErrorsExist()) {
//	  std::vector(edm::ErrorSummaryEntry es = edm::LoggedErrorsSummary();
//	  package_as_product_in_the_event (es);
//	}
//
//
// 25-Aug-2008 mf	Created file.
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <vector>

namespace edm {       

bool EnableLoggedErrorsSummary();
bool DisableLoggedErrorsSummary();
bool FreshErrorsExist();
std::vector<ErrorSummaryEntry> LoggedErrorsSummary();

}        // end of namespace edm


#endif  // MessageLogger_ErrorSummaryEntry_h

