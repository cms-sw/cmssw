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
// $Id: eventsetupplugin_t.cppunit.cc,v 1.3 2005/09/19 08:18:02 chrjones Exp $
//

// system include files
#include <cppunit/extensions/HelperMacros.h>
// user include files
#include "PluginManager/PluginManager.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

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
   dummyFinderPSet.insert(true, "@module_type", edm::Entry(std::string("LoadableDummyFinder"), true));
   dummyFinderPSet.insert(true, "@module_label", edm::Entry(std::string(""), true));
   SourceFactory::get()->addTo(provider, dummyFinderPSet, "RECO", 1, 1);

   
   ComponentDescription descFinder("LoadableDummyFinder","",true);
   std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
   //should not be found since not a provider
   CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

   
   edm::ParameterSet dummyProviderPSet;
   dummyProviderPSet.insert(true, "@module_type", edm::Entry(std::string("LoadableDummyProvider"), true));
   dummyProviderPSet.insert(true, "@module_label", edm::Entry(std::string(""), true));
   ModuleFactory::get()->addTo(provider, dummyProviderPSet, "RECO", 1, 1);

   ComponentDescription desc("LoadableDummyProvider","",false);
   descriptions = provider.proxyProviderDescriptions();
   CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
   CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);

   
   edm::ParameterSet dummySourcePSet;
   dummySourcePSet.insert(true, "@module_type", edm::Entry(std::string("LoadableDummyESSource"), true));
   dummySourcePSet.insert(true, "@module_label", edm::Entry(std::string(""), true));
   SourceFactory::get()->addTo(provider, dummySourcePSet, "RECO", 1, 1);
   
   ComponentDescription descSource("LoadableDummyESSource","",true);
   descriptions = provider.proxyProviderDescriptions();
   CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
   CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
   
}
