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
//

// system include files
#include <cassert>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"


namespace edm {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  //UnscheduledHandler::UnscheduledHandler() {
  //}

  // UnscheduledHandler::UnscheduledHandler(UnscheduledHandler const& rhs)  {
  //    // do actual copying here;
  // }

  UnscheduledHandler::~UnscheduledHandler() {
  }

  //
  // assignment operators
  //
  // UnscheduledHandler const& UnscheduledHandler::operator=(UnscheduledHandler const& rhs) {
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
     CurrentProcessingContext const* old = m_context;
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
     CurrentProcessingContext const* chosen = m_context;
     CurrentProcessingContext temp;
     if(0 != m_context) {
        temp = *m_context;
        temp.setUnscheduledDepth(m_context->unscheduledDepth());
        chosen = &temp;
     }
     UnscheduledHandlerSentry sentry(this, chosen);
     return tryToFillImpl(label, iEvent, *m_setup, chosen);
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
//-------------

   UnscheduledHandlerSentry::UnscheduledHandlerSentry(UnscheduledHandler* iHandler,
                               CurrentProcessingContext const* iContext) :
   m_handler(iHandler),
   m_old(nullptr) {
      if(m_handler) {
	  m_old = iHandler->setCurrentProcessingContext(iContext);
      }
   }

   UnscheduledHandlerSentry::~UnscheduledHandlerSentry() {
      if(m_handler) {
         m_handler->setCurrentProcessingContext(m_old);
      }
   }
}
