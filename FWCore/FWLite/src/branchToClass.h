#ifndef FWLite_BranchToClass_h
#define FWLite_BranchToClass_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BranchToClass
// 
/**\class BranchToClass BranchToClass.h FWCore/FWLite/interface/BranchToClass.h

 Description: Given a TBranch it will return the TClass of the class type stored in the branch

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  2 09:04:02 EDT 2006
// $Id$
//

// system include files

// user include files

// forward declarations
class TClass;
class TBranch;

TClass* branchToClass( const TBranch* iBranch);



#endif
