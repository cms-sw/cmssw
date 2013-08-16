// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventsetup_plugin_t
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 11:01:19 EDT 2005
// Changed:     Viji Sundararajan 28-Jun-2005
//

// system include files
#include <cppunit/extensions/HelperMacros.h>
// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/src/EventSetupsController.h"

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
      if(not edmplugin::PluginManager::isAvailable()) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
     }
      firstTime = false;
   }
}

void testEventsetupplugin::finderTest()

{
   doInit();
   EventSetupsController esController;
   EventSetupProvider provider;
   
   edm::ParameterSet dummyFinderPSet;
   dummyFinderPSet.addParameter("@module_type", std::string("LoadableDummyFinder"));
   dummyFinderPSet.addParameter("@module_label", std::string(""));
   dummyFinderPSet.registerIt();
   SourceFactory::get()->addTo(esController, provider, dummyFinderPSet);

   
   ComponentDescription descFinder("LoadableDummyFinder","",true);
   std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
   //should not be found since not a provider
   CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

   
   edm::ParameterSet dummyProviderPSet;
   dummyProviderPSet.addParameter("@module_type",  std::string("LoadableDummyProvider"));
   dummyProviderPSet.addParameter("@module_label", std::string(""));
   dummyProviderPSet.registerIt();
   ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet);

   ComponentDescription desc("LoadableDummyProvider","",false);
   descriptions = provider.proxyProviderDescriptions();
   CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
   CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);

   
   edm::ParameterSet dummySourcePSet;
   dummySourcePSet.addParameter("@module_type",  std::string("LoadableDummyESSource"));
   dummySourcePSet.addParameter("@module_label", std::string(""));
   dummySourcePSet.registerIt();
   SourceFactory::get()->addTo(esController, provider, dummySourcePSet);
   
   ComponentDescription descSource("LoadableDummyESSource","",true);
   descriptions = provider.proxyProviderDescriptions();
   CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
   CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
   
}
