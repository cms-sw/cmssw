#ifndef FWCore_Framework_UnscheduledHandler_h
#define FWCore_Framework_UnscheduledHandler_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     UnscheduledHandler
//
/**\class UnscheduledHandler UnscheduledHandler.h FWCore/Framework/interface/UnscheduledHandler.h

 Description: Interface to allow handling unscheduled processing

 Usage:
    This class is used internally to the Framework for running the unscheduled case.  It is written as a base class
to keep the EventPrincipal class from having too much 'physical' coupling with the implementation.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb 13 16:26:33 IST 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <string>

// forward declarations
namespace edm {
   class ModuleCallingContext;
   class UnscheduledHandlerSentry;

   class UnscheduledHandler {

   public:
      friend class UnscheduledHandlerSentry;
      UnscheduledHandler(): m_setup(nullptr) {}
      virtual ~UnscheduledHandler();

      UnscheduledHandler(UnscheduledHandler const&) = delete; // Disallow copying and moving
      UnscheduledHandler& operator=(UnscheduledHandler const&) = delete; // Disallow copying and moving

      // ---------- member functions ---------------------------
      ///returns true if found an EDProducer and ran it
      bool tryToFill(std::string const& label,
                     EventPrincipal& iEvent,
                     ModuleCallingContext const* mcc);

      void setEventSetup(EventSetup const& iSetup) {
         m_setup = &iSetup;
      }
   private:

      virtual bool tryToFillImpl(std::string const&,
                                 EventPrincipal&,
                                 EventSetup const&,
                                 ModuleCallingContext const* mcc) = 0;
      // ---------- member data --------------------------------
      EventSetup const* m_setup;
   };
}
#endif
