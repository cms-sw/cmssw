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
// $Id: Context.cc,v 1.6 2010/01/21 21:01:30 amraktad Exp $
//

// system include files

// user include files
#include "TEveTrackPropagator.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWMagField.h"

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
   m_colorManager(iColorM),
   m_propagator(0),
   m_magField(0)
{
   m_magField = new FWMagField();

   m_propagator = new TEveTrackPropagator();
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);
   m_propagator->SetMagFieldObj(m_magField);
   m_propagator->IncRefCount();
}


Context::~Context()
{
}

//
// static member functions
//


//
// static data member definitions
//
//
// const member functions
//
//
// member functions
