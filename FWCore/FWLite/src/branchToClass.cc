// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BranchToClass
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  2 09:04:04 EDT 2006
//

// system include files
class TBranch;
class TClass;
#include "TBranchBrowsable.h"

// user include files
#include "FWCore/FWLite/src/branchToClass.h"

namespace {
  class BranchToClass : private TVirtualBranchBrowsable {
  public:
    static TClass* doit(const TBranch* iBranch);

  private:
    ///NOTE: do not call this, it is only here because ROOT demands it
    BranchToClass() = delete;
  };

  TClass* BranchToClass::doit(const TBranch* iBranch) {
    TClass* contained = nullptr;
    TClass* type = TVirtualBranchBrowsable::GetCollectionContainedType(iBranch, nullptr, contained);
    if (type == nullptr) {
      type = contained;
    }
    return type;
  }

}  // namespace

TClass* branchToClass(const TBranch* iBranch) { return BranchToClass::doit(iBranch); }
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
