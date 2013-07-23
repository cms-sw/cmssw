// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginfactorymanager_t
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:38:29 EDT 2007
// $Id: pluginfactorymanager_t.cc,v 1.2 2007/04/12 12:51:13 wmtan Exp $
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>
#include <iostream>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"

class TestPluginFactoryManager : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestPluginFactoryManager);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();
public:
    void test();
    void setUp() {}
    void tearDown() {}
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestPluginFactoryManager);

class DummyTestPlugin : public edmplugin::PluginFactoryBase {
public:
  DummyTestPlugin(const std::string& iName): name_(iName) {
    finishedConstruction();
  }
  const std::string& category() const {return name_;}
  std::vector<edmplugin::PluginInfo> available() const {
    return std::vector<edmplugin::PluginInfo>();
  }
  const std::string name_;
};

struct Catcher {
  std::string lastSeen_;
  
  void catchIt(const edmplugin::PluginFactoryBase* iFactory) {
    lastSeen_=iFactory->category();
  }
};

void
TestPluginFactoryManager::test()
{
  using namespace edmplugin;
  PluginFactoryManager& pfm = *(PluginFactoryManager::get());
  CPPUNIT_ASSERT(pfm.begin()==pfm.end());
  
  Catcher catcher;
  pfm.newFactory_.connect(boost::bind(boost::mem_fn(&Catcher::catchIt),&catcher,_1));
  
  DummyTestPlugin one("one");
  CPPUNIT_ASSERT((pfm.begin()!=pfm.end()));
  CPPUNIT_ASSERT(catcher.lastSeen_ == std::string("one"));
  
}
