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
// $Id: branchToClass.cc,v 1.2 2007/06/14 21:03:37 wmtan Exp $
//

// system include files
class TBranch;
class TClass;
#include "TBranchBrowsable.h"

// user include files
#include "FWCore/FWLite/src/branchToClass.h"

namespace {
class BranchToClass : private TVirtualBranchBrowsable
{
  
public:
  static TClass* doit( const TBranch* iBranch );
  
private:
  ///NOTE: do not call this, it is only here because ROOT demands it
  BranchToClass();
};

TClass*
BranchToClass::doit( const TBranch* iBranch )
{
  TClass* contained = 0;
  TClass* type = TVirtualBranchBrowsable::GetCollectionContainedType(iBranch,0,contained);
  if( type == 0) {
    type = contained;
  }
  return type;  
}

}

TClass*
branchToClass(const TBranch* iBranch)
{
  return BranchToClass::doit(iBranch);
}
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
