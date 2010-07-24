#ifndef FWCore_Utilities_UseReflex_h
#define FWCore_Utilities_UseReflex_h

// The original purpose of this header was to isolate the changes that need to be made 
// when migrating from ROOT 5.18 to ROOT 5.21 due to the fact that the 
// namespace ROOT::Reflex in 5.18 is simply Reflex: in 5.21.

// However, the conditional code for ROOT 5.19 and prior releases (now obsolete in CMSSW)
// has been removed, so this header is just a convenience for files that use reflex.

#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"

namespace Reflex {
  class Base;
  class Member;
  class Object;
  class PropertyList;
  class SharedLibrary;
  class Type;
  class TypeTemplate;
  inline
  std::ostream& operator<<(std::ostream& os, Type const& t) {
    os << t.Name();
    return os;
  } 
  inline
  std::ostream& operator<<(std::ostream& os, TypeTemplate const& tt) {
    os << tt.Name();
    return os;
  }
}
#endif
