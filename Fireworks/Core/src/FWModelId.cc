// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelId
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Mar  5 11:00:48 EST 2008
// $Id: FWModelId.cc,v 1.4 2009/01/23 21:35:43 amraktad Exp $
//

// system include files

// user include files
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
/*
   FWModelId::FWModelId()
   {
   }
 */
// FWModelId::FWModelId(const FWModelId& rhs)
// {
//    // do actual copying here;
// }

/*
   FWModelId::~FWModelId()
   {
   }
 */

//
// assignment operators
//
// const FWModelId& FWModelId::operator=(const FWModelId& rhs)
// {
//   //An exception safe implementation is
//   FWModelId temp(rhs);
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
void
FWModelId::unselect() const {
   if(m_item) {m_item->unselect(m_index);}
}
void
FWModelId::select() const {
   if(m_item) {m_item->select(m_index);}
}

void
FWModelId::toggleSelect() const {
   if(m_item) {m_item->toggleSelect(m_index);}
}

//
// static member functions
//
