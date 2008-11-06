// -*- C++ -*-
//
// Package:     Core
// Class  :     Context
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 30 14:57:12 EDT 2008
// $Id: Context.cc,v 1.1 2008/10/21 19:06:56 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/Context.h"

using namespace fireworks;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Context::Context(FWModelChangeManager* iCM,
                 FWSelectionManager* iSM):
m_changeManager(iCM),
m_selectionManager(iSM)
{
}

// Context::Context(const Context& rhs)
// {
//    // do actual copying here;
// }

//Context::~Context()
//{
//}

//
// assignment operators
//
// const Context& Context::operator=(const Context& rhs)
// {
//   //An exception safe implementation is
//   Context temp(rhs);
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
