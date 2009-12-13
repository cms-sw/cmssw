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
// $Id: FWViewContextMenuHandlerBase.cc,v 1.1 2009/11/02 23:59:49 chrjones Exp $
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
m_handler(&iHandler),m_lastIndex(0) {}
   
int 
FWViewContextMenuHandlerBase::MenuEntryAdder::addEntry(const char* iEntryName)
{
   m_handler->addViewEntry(iEntryName,m_lastIndex);
   return m_lastIndex++;
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
FWViewContextMenuHandlerBase::addTo(FWModelContextMenuHandler& iHandler)
{
   MenuEntryAdder adder(iHandler);
   init(adder);
}
