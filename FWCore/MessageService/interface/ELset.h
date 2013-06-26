#ifndef FWCore_MessageService_ELset_h
#define FWCore_MessageService_ELset_h


// ----------------------------------------------------------------------
//
// ELset.h     Provides a set class with the semantics of std::set.
//              Customizers may substitute for this class to provide either
//              a set with a different allocator, or whatever else.
//
// The following elements of set semantics are relied upon:
//      insert()
//	clear()
//	find() which returns an iterator which may or may not be .end()
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ELstring.h"

#include <set>


namespace edm {       
namespace service {       


struct ELsetS {
  std::string s;
  ELsetS (const std::string & ss) : s(ss) {}
  bool operator< (const ELsetS & t) const { return (s<t.s); }
};

typedef std::set<ELsetS> ELset_string;

// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELset_h
