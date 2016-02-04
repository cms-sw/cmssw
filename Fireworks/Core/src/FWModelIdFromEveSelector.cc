// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelIdFromEveSelector
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 28 11:44:16 CET 2009
// $Id: FWModelIdFromEveSelector.cc,v 1.1 2009/10/28 14:36:58 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWModelIdFromEveSelector.h"
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
//FWModelIdFromEveSelector::FWModelIdFromEveSelector()
//{
//}

// FWModelIdFromEveSelector::FWModelIdFromEveSelector(const FWModelIdFromEveSelector& rhs)
// {
//    // do actual copying here;
// }

//FWModelIdFromEveSelector::~FWModelIdFromEveSelector()
//{
//}

//
// assignment operators
//
// const FWModelIdFromEveSelector& FWModelIdFromEveSelector::operator=(const FWModelIdFromEveSelector& rhs)
// {
//   //An exception safe implementation is
//   FWModelIdFromEveSelector temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWModelIdFromEveSelector::doSelect()
{
   if( not m_id.item()->modelInfo(m_id.index()).isSelected() ) {
      m_id.select();
   }
}

void 
FWModelIdFromEveSelector::doUnselect()
{
   if( m_id.item()->modelInfo(m_id.index()).isSelected() ) {
      m_id.unselect();
   }
}

//
// const member functions
//

//
// static member functions
//
