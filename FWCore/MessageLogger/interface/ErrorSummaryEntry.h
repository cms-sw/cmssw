#ifndef MessageLogger_ErrorSummaryEntry_h
#define MessageLogger_ErrorSummaryEntry_h

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"

#include <string>

// ----------------------------------------------------------------------
//
// ErrorSummaryEntry.h - Structure to hold summary of a warning or error
//		    message issued in an event.  
//
//   Usage:  
//	if (edm::FreshErrorsExist()) {
//	  std::vector(edm::ErrorSummaryEntry es = edm::LoggedErrorsSummary();
//	  package_as_product_in_the_event (es);
//	}
//
//   edm::ErrorSummaryEntry is a very simple struct, containing only strings
//   and an int; it is very suitable for inclusion in the event.
//
//   Unusual C++ practice warning:
//     edm::ErrorSummaryEntry is a simple struct; its members are public and
//     are accessed directly rather than through accessor functions.  Thus it
//     **is reasonable** in this case to treat the code defining the
//     edm::ErrorSummaryEntry as documentation of how to use it.
//
// 20-Aug-2008 mf	Created file.
// 
// 22-Jun-2009 mf	Added severity to the structure.  This adds just one
//			integer to the memory used.
//
// ----------------------------------------------------------------------

namespace edm {       

struct ErrorSummaryEntry 
{
  std::string     category;
  std::string     module;
  ELseverityLevel severity;
  unsigned int    count;
  ErrorSummaryEntry(std::string const & cat, std::string const & mod, 
  		    ELseverityLevel sev, unsigned int cnt = 0) 
	: category(cat)
	, module  (mod)
	, severity(sev)
	, count(cnt) {}
  ErrorSummaryEntry() : category(), module(), severity(), count(0) {}
  bool operator< (ErrorSummaryEntry const & rhs) const {
    if (category < rhs.category) return true;
    if (category > rhs.category) return false; 
    if (module   < rhs.module)   return true;
    if (module   > rhs.module)   return false; 
    if (severity < rhs.severity) return true;
    if (severity > rhs.severity) return false; 
    if (count    < rhs.count)    return true;
    return false; 
  }
  bool operator== (ErrorSummaryEntry const & rhs) const {
    return ( (category < rhs.category) && (module < rhs.module)
    	  && (severity < rhs.severity) && (count  < rhs.count)  );
  }
};

}        // end of namespace edm


#endif  // MessageLogger_ErrorSummaryEntry_h

