// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginfactory_t
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:38:29 EDT 2007
// $Id: pluginfactory_t.cc,v 1.6 2008/01/22 22:10:36 wmtan Exp $
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#define private public
#include "FWCore/PluginManager/interface/PluginFactory.h"

class TestPluginFactory : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestPluginFactory);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST(testTry);
  CPPUNIT_TEST_SUITE_END();
public:
    void test();
    void testTry();
    void setUp() {   
      if (!alreadySetup_) {
        alreadySetup_=true;
        edmplugin::PluginManager::configure(edmplugin::standard::config());
      }
    }
    void tearDown() {}
    static bool alreadySetup_;
};

bool TestPluginFactory::alreadySetup_ = false;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestPluginFactory);

namespace edmplugintest {
  struct DummyBase {};
  
  struct Dummy: public DummyBase {};
}

typedef edmplugin::PluginFactory<edmplugintest::DummyBase*(void)> FactoryType;
EDM_REGISTER_PLUGINFACTORY(FactoryType,"Test Dummy");

DEFINE_EDM_PLUGIN(FactoryType,edmplugintest::Dummy,"Dummy");

void
TestPluginFactory::test()
{
  using namespace edmplugin;
  
  std::auto_ptr<edmplugintest::DummyBase> p(FactoryType::get()->create("Dummy"));
  CPPUNIT_ASSERT(0 != p.get());
}  

void
TestPluginFactory::testTry()
{
  using namespace edmplugin;
  CPPUNIT_ASSERT(0 == FactoryType::get()->tryToCreate("ThisDoesNotExist"));
  
  std::auto_ptr<edmplugintest::DummyBase> p(FactoryType::get()->tryToCreate("Dummy"));
  CPPUNIT_ASSERT(0 != p.get());
}  
