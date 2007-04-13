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
// $Id: pluginfactory_t.cc,v 1.2 2007/04/12 12:51:13 wmtan Exp $
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <sstream>

// user include files
#define private public
#include "FWCore/PluginManager/interface/PluginFactory.h"

class TestPluginFactory : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestPluginFactory);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();
public:
    void test();
    void setUp() {}
    void tearDown() {}
};

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
