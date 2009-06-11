// -*- C++ -*-
//
// Package:     Framework
// Class  :     UnscheduledHandler
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Oct 13 13:58:07 CEST 2008
// $Id$
//

// system include files
#include <iostream>

// user include files
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//UnscheduledHandler::UnscheduledHandler()
//{
//}

// UnscheduledHandler::UnscheduledHandler(const UnscheduledHandler& rhs)
// {
//    // do actual copying here;
// }

UnscheduledHandler::~UnscheduledHandler()
{
}

//
// assignment operators
//
// const UnscheduledHandler& UnscheduledHandler::operator=(const UnscheduledHandler& rhs)
// {
//   //An exception safe implementation is
//   UnscheduledHandler temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
CurrentProcessingContext const* 
UnscheduledHandler::setCurrentProcessingContext(CurrentProcessingContext const* iContext) {
   const CurrentProcessingContext* old = m_context;
   m_context = iContext;
   return old;
}

/*
void 
UnscheduledHandler::popCurrentProcessingContext() {
}
*/
 
bool 
UnscheduledHandler::tryToFill(std::string const& label,
                              EventPrincipal& iEvent) {
   assert(m_setup);
   const CurrentProcessingContext* chosen = m_context;
   CurrentProcessingContext temp;
   if(0!=m_context) {
      temp = *m_context;
      temp.setUnscheduledDepth(m_context->unscheduledDepth());
      chosen = &temp;
   }
   UnscheduledHandlerSentry sentry(this,chosen);
   return tryToFillImpl(label, iEvent, *m_setup,chosen);
}

//
// const member functions
//

//
// static member functions
//
