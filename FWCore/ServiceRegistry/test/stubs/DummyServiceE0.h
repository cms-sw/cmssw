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
// $Id$
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace testserviceregistry {

  // The single extra letter in each of these typenames is intentionally
  // random.  These are used in a test of the ordering of service
  // calls, construction and destruction.  This previously
  // depended on the type and I want to be sure this is not
  // true anymore.

   class DummyServiceE0
   {
   public:
      DummyServiceE0(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceE0();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceA1
   {
   public:
      DummyServiceA1(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceA1();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceD2
   {
   public:
      DummyServiceD2(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceD2();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceB3
   {
   public:
      DummyServiceB3(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceB3();
      void postBeginJob();
      void postEndJob();
   };

   class DummyServiceC4
   {
   public:
      DummyServiceC4(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~DummyServiceC4();
      void postBeginJob();
      void postEndJob();
   };
}

#endif
