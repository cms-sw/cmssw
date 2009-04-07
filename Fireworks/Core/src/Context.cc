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
// $Id: Context.cc,v 1.4 2009/01/23 21:35:42 amraktad Exp $
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
                 FWSelectionManager* iSM,
                 FWEventItemsManager* iEM,
                 FWColorManager* iColorM
) :
   m_changeManager(iCM),
   m_selectionManager(iSM),
   m_eventItemsManager(iEM),
   m_colorManager(iColorM)
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
