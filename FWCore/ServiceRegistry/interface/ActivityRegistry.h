#ifndef ServiceRegistry_ActivityRegistry_h
#define ServiceRegistry_ActivityRegistry_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ActivityRegistry
// 
/**\class ActivityRegistry ActivityRegistry.h FWCore/ServiceRegistry/interface/ActivityRegistry.h

 Description: Registry holding the boost::signals that Services can subscribe to

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:53:09 EDT 2005
// $Id$
//

// system include files
#include "boost/signal.hpp"

// user include files

// forward declarations
namespace edm {
   struct ActivityRegistry
   {
      ActivityRegistry() {}
      // ---------- signals ------------------------------------

      
      // ---------- member functions ---------------------------

      ///forwards our signals to slots connected to iOther
      void connect(ActivityRegistry& iOther);
      
   private:
      ActivityRegistry(const ActivityRegistry&); // stop default

      const ActivityRegistry& operator=(const ActivityRegistry&); // stop default

      // ---------- member data --------------------------------

   };
}
#endif
