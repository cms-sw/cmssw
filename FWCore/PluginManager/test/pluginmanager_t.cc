// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginmanager_t
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:38:29 EDT 2007
// $Id: pluginmanager_t.cc,v 1.5 2008/01/22 22:10:38 wmtan Exp $
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/PluginManager/test/DummyFactory.h"

class TestPluginManager : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestPluginManager);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();
public:
    void test();
    void setUp() {}
    void tearDown() {}
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestPluginManager);

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
/*
struct Catcher {
  std::string lastSeen_;
  
  void catchIt(const edmplugin::PluginFactoryBase* iFactory) {
    lastSeen_=iFactory->category();
  }
};
*/

namespace testedmplugin {
  struct DummyThree: public DummyBase {
    int value() const {
      return 3;
    }
  };
}

DEFINE_EDM_PLUGIN(testedmplugin::DummyFactory,testedmplugin::DummyThree,"DummyThree");


void
TestPluginManager::test()
{
  using namespace edmplugin;
  using namespace testedmplugin;
  CPPUNIT_ASSERT_THROW(PluginManager::get(), cms::Exception );

  PluginManager::Config config;
  CPPUNIT_ASSERT_THROW(PluginManager::configure(config), cms::Exception );
  
  const char* path = getenv("LD_LIBRARY_PATH");
  CPPUNIT_ASSERT(0 != path);
  
  std::string spath(path? path: "");
  std::string::size_type last=0;
  std::string::size_type i=0;
  std::vector<std::string> paths;
  while( (i=spath.find_first_of(':',last))!=std::string::npos) {
    paths.push_back(spath.substr(last,i-last));
    last = i+1;
    std::cout <<paths.back()<<std::endl;
  }
  paths.push_back(spath.substr(last,std::string::npos));
  config.searchPath(paths);
  
  edmplugin::PluginManager& db = edmplugin::PluginManager::configure(config);

  std::auto_ptr<DummyBase> ptr(DummyFactory::get()->create("DummyOne"));
  CPPUNIT_ASSERT(1==ptr->value());
  CPPUNIT_ASSERT(db.loadableFor("Test Dummy", "DummyThree") == "static");
  std::auto_ptr<DummyBase> ptr2(DummyFactory::get()->create("DummyThree"));
  CPPUNIT_ASSERT(3==ptr2->value());
 
  CPPUNIT_ASSERT_THROW(DummyFactory::get()->create("DoesNotExist"), cms::Exception);
  
}
