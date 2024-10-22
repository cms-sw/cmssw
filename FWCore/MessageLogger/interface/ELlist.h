#ifndef MessageLogger_ELlist_h
#define MessageLogger_ELlist_h

// ----------------------------------------------------------------------
//
// ELlist.h     Provides a list class with the semantics of std::list.
//              Customizers may substitute for this class to provide either
//              a list with a different allocator, or whatever else.
//
//	We typedef an individual type for each of these lists since
//	the syntax
//		typedef list ELlist;
//		ELlist<ELdestination> sinks;
//	may or may not be valid C++, and if valid probably won't work
//	everywhere.
//
// The following elements of list semantics are relied upon:
//      push_back()             ELadminstrator
//
//
// ----------------------------------------------------------------------

#include <list>
#include <string>

namespace edm {

  // ----------------------------------------------------------------------

  class ELdestination;
  typedef std::list<ELdestination *> ELlist_dest;

  typedef std::list<std::string> ELlist_string;

  // ----------------------------------------------------------------------

}  // end of namespace edm

#endif  // MessageLogger_ELlist_h
