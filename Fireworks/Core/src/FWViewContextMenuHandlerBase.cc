// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContextMenuHandlerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon Nov  2 13:46:48 CST 2009
// $Id: FWViewContextMenuHandlerBase.cc,v 1.4 2011/03/25 18:02:46 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
FWViewContextMenuHandlerBase::MenuEntryAdder::MenuEntryAdder(FWModelContextMenuHandler& iHandler):
m_handler(&iHandler){}
   
int 
FWViewContextMenuHandlerBase::MenuEntryAdder::addEntry(const char* iEntryName, int idx, bool enabled)
{
   m_handler->addViewEntry(iEntryName, idx, enabled);
   return idx;
}


//
// constructors and destructor
//
FWViewContextMenuHandlerBase::FWViewContextMenuHandlerBase()
{
}

// FWViewContextMenuHandlerBase::FWViewContextMenuHandlerBase(const FWViewContextMenuHandlerBase& rhs)
// {
//    // do actual copying here;
// }

FWViewContextMenuHandlerBase::~FWViewContextMenuHandlerBase()
{
}

//
// assignment operators
//
// const FWViewContextMenuHandlerBase& FWViewContextMenuHandlerBase::operator=(const FWViewContextMenuHandlerBase& rhs)
// {
//   //An exception safe implementation is
//   FWViewContextMenuHandlerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWViewContextMenuHandlerBase::addTo(FWModelContextMenuHandler& iHandle, const FWModelId &id)
{
   MenuEntryAdder adder(iHandle);
   init(adder, id);
}
