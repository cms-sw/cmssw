#ifndef MessageLogger_ErrorSummaryEntry_h
#define MessageLogger_ErrorSummaryEntry_h

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
// ----------------------------------------------------------------------

namespace edm {       

struct ErrorSummaryEntry 
{
  std::string category;
  std::string module;
  unsigned int count;
};

}        // end of namespace edm


#endif  // MessageLogger_ErrorSummaryEntry_h

