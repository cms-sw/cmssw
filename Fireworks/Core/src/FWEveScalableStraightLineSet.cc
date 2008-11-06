// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveScalableStraightLineSet
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Jul  3 16:25:33 EDT 2008
// $Id: FWEveScalableStraightLineSet.cc,v 1.1 2008/07/04 23:39:30 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveScalableStraightLineSet::FWEveScalableStraightLineSet(const Text_t* iName,
                                                           const Text_t* iTitle):
TEveScalableStraightLineSet(iName,iTitle)
{
}

// FWEveScalableStraightLineSet::FWEveScalableStraightLineSet(const FWEveScalableStraightLineSet& rhs)
// {
//    // do actual copying here;
// }
/*
FWEveScalableStraightLineSet::~FWEveScalableStraightLineSet()
{
}
*/
//
// assignment operators
//
// const FWEveScalableStraightLineSet& FWEveScalableStraightLineSet::operator=(const FWEveScalableStraightLineSet& rhs)
// {
//   //An exception safe implementation is
//   FWEveScalableStraightLineSet temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWEveScalableStraightLineSet::setScale(float iScale)
{
   SetScale(iScale);
}
//
// const member functions
//

//
// static member functions
//
