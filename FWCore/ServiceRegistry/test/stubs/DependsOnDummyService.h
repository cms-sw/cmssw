#ifndef ServiceRegistry_test_DependsOnDummyService_h
#define ServiceRegistry_test_DependsOnDummyService_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     DependsOnDummyService
// 
/**\class DependsOnDummyService DependsOnDummyService.h FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:51:59 EDT 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
namespace testserviceregistry {
   class DependsOnDummyService
   {

   public:
      DependsOnDummyService();
      virtual ~DependsOnDummyService();

      // ---------- const member functions ---------------------
      int value() const { return value_; }
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      DependsOnDummyService(const DependsOnDummyService&); // stop default

      const DependsOnDummyService& operator=(const DependsOnDummyService&); // stop default

      // ---------- member data --------------------------------
      int value_;
   };
}

#endif
