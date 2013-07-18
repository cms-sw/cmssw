#ifndef ServiceRegistry_test_DummyServiceE0_h
#define ServiceRegistry_test_DummyServiceE0_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     DummyServiceE0
// 
/**\class DummyServiceE0 DummyServiceE0.h FWCore/ServiceRegistry/test/stubs/DummyServiceE0.h

 Description: Used for tests of ServicesManager

 Usage:
    <usage>

*/
//
// Original Author:  David Dagenhart
// $Id: DummyServiceE0.h,v 1.1 2007/02/14 20:45:13 wdd Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class ConfigurationDescriptions;
}

namespace testserviceregistry {

  // The single extra letter in each of these typenames is intentionally
  // random.  These are used in a test of the ordering of service
  // calls, construction and destruction.  This previously
  // depended on the type and I want to be sure this is not
  // true anymore.

   class DummyServiceBase {
   public:
     static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
   };

   class DummyServiceE0 : public DummyServiceBase
   {
   public:
      DummyServiceE0(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceE0();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceA1 : public DummyServiceBase
   {
   public:
      DummyServiceA1(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceA1();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceD2 : public DummyServiceBase
   {
   public:
      DummyServiceD2(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceD2();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceB3 : public DummyServiceBase
   {
   public:
      DummyServiceB3(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceB3();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceC4 : public DummyServiceBase
   {
   public:
      DummyServiceC4(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceC4();
      void postBeginJob();
      void postEndJob();
   };
}

#endif
