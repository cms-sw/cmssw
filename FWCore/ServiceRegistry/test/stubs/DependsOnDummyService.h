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
//

namespace edm {
   class ConfigurationDescriptions;
}

namespace testserviceregistry {
   class DependsOnDummyService
   {

   public:
      DependsOnDummyService();
      virtual ~DependsOnDummyService();

      int value() const { return value_; }
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      DependsOnDummyService(const DependsOnDummyService&); // stop default
      const DependsOnDummyService& operator=(const DependsOnDummyService&); // stop default

      int value_;
   };
}
#endif
