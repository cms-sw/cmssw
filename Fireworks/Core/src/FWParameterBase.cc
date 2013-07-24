// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Feb 23 13:36:24 EST 2008
// $Id: FWParameterBase.cc,v 1.4 2012/02/22 03:46:00 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWParameterizable.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

FWParameterBase::FWParameterBase(FWParameterizable* iParent,const std::string& iName) :
   m_name(iName)
{
   if (0 != iParent)
   {
      iParent->add(this);
   }
}

// FWParameterBase::FWParameterBase(const FWParameterBase& rhs)
// {
//    // do actual copying here;
// }

FWParameterBase::~FWParameterBase()
{
}

//
// assignment operators
//
// const FWParameterBase& FWParameterBase::operator=(const FWParameterBase& rhs)
// {
//   //An exception safe implementation is
//   FWParameterBase temp(rhs);
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
