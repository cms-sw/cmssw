#ifndef MessageLogger_LoggedErrorsSummaryy_h
#define MessageLogger_LoggedErrorsSummaryEntry_h

// ----------------------------------------------------------------------
//
// LoggedErrorsSummary.h - Methods to obtain summary of warning and error
//		    messages issued in an event.  
//
//   Usage:  
//	EnableLoggedErrorsSummary();
//	then per event:
//	some_code_that_might_issue_messages();
//	if (edm::FreshErrorsExist()) {
//	  std::vector<edm::ErrorSummaryEntry> es = edm::LoggedErrorsSummary();
//	  package_as_product_in_the_event (es);
//	}
//
//  The above gives warnings and errors; use LoggedErrorsOnlySummary() for
//  just errors.
//
//  Note:  This goes by severity level.  Thus a LogImportant message, though
//         not intended to convey a problematic error, will appear in the
//         summary.  Also, Absolute and System messages are logged.

//
//  void
//  package_as_product_in_the_event(std::vector<edm::ErrorSummaryEntry> const& es)
//  {
//    // This example shows how to save just errors and not warnings
//    std::vector<edm::ErrorSummaryEntry> errs;
//    std::vector<edm::ErrorSummaryEntry>::const_iterator end = es.end();
//    for (std::vector<edm::ErrorSummaryEntry>::const_iterator i = es.begin();
//                                                        i != end; ++i) { 
//      if ( i->severity >= edm:ELerror ) errs.push_back(*i);
//    }
//    place_into_event(errs);
//  }
//
// 25-Aug-2008 mf	Created file.
//
// 22-Jun-2009 mf	Added LoggedErrorsOnlySummary()
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <vector>

namespace edm {       

bool EnableLoggedErrorsSummary();
bool DisableLoggedErrorsSummary();
bool FreshErrorsExist();
std::vector<ErrorSummaryEntry> LoggedErrorsSummary();     // Errors and Warnings
std::vector<ErrorSummaryEntry> LoggedErrorsOnlySummary(); // Errors only

}        // end of namespace edm


#endif  // MessageLogger_ErrorSummaryEntry_h

