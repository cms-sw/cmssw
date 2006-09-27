#ifndef Framework_UnscheduledHandler_h
#define Framework_UnscheduledHandler_h
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
// $Id: UnscheduledHandler.h,v 1.1 2006/03/05 21:40:25 chrjones Exp $
//

// system include files
#include <cassert>

// user include files
#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

// forward declarations
namespace edm {
   
   class UnscheduledHandler
{

   public:
   UnscheduledHandler(): m_setup(0) {}
      virtual ~UnscheduledHandler() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      ///returns true if found an EDProducer and ran it
      virtual bool tryToFill(const Provenance& iProd,
                             EventPrincipal& iEvent) {
         assert(m_setup);
         return tryToFillImpl(iProd, iEvent, *m_setup);
      }
      void setEventSetup(const EventSetup& iSetup) {
         m_setup = &iSetup;
      }
   private:
      UnscheduledHandler(const UnscheduledHandler&); // stop default

      const UnscheduledHandler& operator=(const UnscheduledHandler&); // stop default

      virtual bool tryToFillImpl(const Provenance&,
                                 EventPrincipal&,
                                 const EventSetup& ) = 0;
      // ---------- member data --------------------------------
      const EventSetup* m_setup;
};
}

#endif
