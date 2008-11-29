#ifndef FWCore_Utilities_UseReflex_h
#define FWCore_Utilities_UseReflex_h

// The purpose of this header is to isolate the changes that need to be made when migrating
// from ROOT 5.18 to ROOT 5.21 due to the fact that the namespace ROOT::Reflex in 5.18
// is simply Reflex: in 5.21.

// This header may be used as is in 5.18.
// To use this header, do the following, and include this header in any files where any of these apply:
// 1) In any file where ROOT::Reflex is explicitly used, replace ROOT::Reflex with Reflex.
// 2) In any file using "namespace ROOT { namespace Reflex { ... } }", remove this.
//      Any declarations in ... not duplicated in this header would need to be added below.
// 3) In any file containing "using nameapace ROOT::Reflex", remove this directive, and add "Reflex::" where necessary to compile.

// Then, to migrate to 5.21, just remove the three lines indicated below.
// An #ifdef may be used if desired.

#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"

//namespace ROOT { // Remove this line for ROOT 5.21.
namespace Reflex {
  class Base;
  class Member;
  class Object;
  class PropertyList;
  class SharedLibrary;
  class Type;
  class TypeTemplate;
  inline
  std::ostream& operator<< (std::ostream& os, Type const& t) {
    os << t.Name();
    return os;
  } 
  inline
  std::ostream& operator<< (std::ostream& os, TypeTemplate const& tt) {
    os << tt.Name();
    return os;
  }
}
//} // Remove this line for ROOT 5.21.

//using namespace ROOT; // Remove this line for ROOT 5.21.

#endif
