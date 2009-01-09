// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:56 EST 2009
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDetailViewBase::FWDetailViewBase(const std::type_info& iInfo):
m_helper(iInfo)
{
}

// FWDetailViewBase::FWDetailViewBase(const FWDetailViewBase& rhs)
// {
//    // do actual copying here;
// }

FWDetailViewBase::~FWDetailViewBase()
{
}

//
// assignment operators
//
// const FWDetailViewBase& FWDetailViewBase::operator=(const FWDetailViewBase& rhs)
// {
//   //An exception safe implementation is
//   FWDetailViewBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
TEveElement* 
FWDetailViewBase::build (const FWModelId & iID)
{
   m_helper.itemChanged(iID.item());
   return build(iID, m_helper.offsetObject(iID.item()->modelData(iID.index())));
}

//
// const member functions
//

//
// static member functions
//
