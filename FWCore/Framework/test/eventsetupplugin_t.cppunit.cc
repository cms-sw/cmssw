// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     eventsetup_plugin_t
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 11:01:19 EDT 2005
// Changed:     Viji Sundararajan 28-Jun-2005
// $Id: eventsetup_plugin_t.cpp,v 1.3 2005/06/23 22:01:31 wmtan Exp $
//

// system include files
#include <cppunit/extensions/HelperMacros.h>
// user include files
#include "PluginManager/PluginManager.h"
#include "FWCore/CoreFramework/interface/SourceFactory.h"
#include "FWCore/CoreFramework/interface/ModuleFactory.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"

using namespace edm::eventsetup;

class testEventsetupplugin: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testEventsetupplugin);

CPPUNIT_TEST(finderTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void finderTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetupplugin);

static void doInit() {
   static bool firstTime=true;
   if(firstTime) {
      seal::PluginManager::get()->initialise();
      firstTime = false;
   }
}

void testEventsetupplugin::finderTest()

{
   doInit();
   
   EventSetupProvider provider;
   
   edm::ParameterSet dummyFinderPSet;
   dummyFinderPSet.insert(true, "module_type", edm::Entry(std::string("LoadableDummyFinder"), true));
   dummyFinderPSet.insert(true, "module_label", edm::Entry(std::string(""), true));
   SourceFactory::get()->addTo(provider, dummyFinderPSet, "RECO", 1, 1);
   
   edm::ParameterSet dummyProviderPSet;
   dummyProviderPSet.insert(true, "module_type", edm::Entry(std::string("LoadableDummyProvider"), true));
   dummyProviderPSet.insert(true, "module_label", edm::Entry(std::string(""), true));
   ModuleFactory::get()->addTo(provider, dummyProviderPSet, "RECO", 1, 1);
   
}
