#ifndef ServiceRegistry_test_DummyService_h
#define ServiceRegistry_test_DummyService_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     DummyService
// 
/**\class DummyService DummyService.h FWCore/ServiceRegistry/test/stubs/DummyService.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:51:59 EDT 2005
// $Id: DummyService.h,v 1.2 2010/01/19 22:37:05 wdd Exp $
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class ConfigurationDescriptions;
}

// forward declarations
namespace testserviceregistry {
   class DummyService
   {

   public:
      DummyService(const edm::ParameterSet&,edm::ActivityRegistry&);
      virtual ~DummyService();

      // ---------- const member functions ---------------------
      int value() const { return value_; }
      
      bool beginJobCalled() const {return beginJobCalled_;}

      // ---------- static member functions --------------------

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      // ---------- member functions ---------------------------

   private:
      void doOnBeginJob() { beginJobCalled_=true;}
      DummyService(const DummyService&); // stop default

      const DummyService& operator=(const DummyService&); // stop default

      // ---------- member data --------------------------------
      int value_;
      bool beginJobCalled_;
   };
}

#endif
