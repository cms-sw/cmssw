// ----------------------------------------------------------------------
//
// ELstring.cc  Provides a string class with the semantics of std::string.
//              Customizers may substitute for this class to provide either
//              a string with a different allocator, or whatever else.
//
// The elements of string semantics which are relied upon are listed
// in doc/ELstring.semantics
//
// History:
//   15-Nov-2001  WEB  Inserted missing #include <cctype>
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ELstring.h"
#include <cctype>
#include <cstring>

namespace edm {

  bool eq(const ELstring& s1, const ELstring s2) { return s1 == s2; }  // eq()

}  // end of namespace edm  */
