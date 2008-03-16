// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 14:43:19 EST 2008
// $Id: FWViewBase.cc,v 1.1 2008/02/21 20:31:24 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewBase::FWViewBase(unsigned int iVersion):
FWConfigurableParameterizable(iVersion)
{
}

// FWViewBase::FWViewBase(const FWViewBase& rhs)
// {
//    // do actual copying here;
// }

FWViewBase::~FWViewBase()
{
}

//
// assignment operators
//
// const FWViewBase& FWViewBase::operator=(const FWViewBase& rhs)
// {
//   //An exception safe implementation is
//   FWViewBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
