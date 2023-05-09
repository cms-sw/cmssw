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
#include "cppunit/extensions/HelperMacros.h"
// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "DummyFinder.h"
#include "DummyProxyProvider.h"
#include "TestTypeResolvers.h"

using namespace edm::eventsetup;

namespace {
  edm::ActivityRegistry activityRegistry;
}

namespace edm::test {
  namespace other {
    class LoadableDummyFinderA : public DummyFinder {
    public:
      LoadableDummyFinderA(const edm::ParameterSet&) { ++count_; }
      static int count_;
    };
    int LoadableDummyFinderA::count_ = 0;

    class LoadableDummyProviderA : public edm::eventsetup::test::DummyProxyProvider {
    public:
      LoadableDummyProviderA(const edm::ParameterSet& iPSet)
          : DummyProxyProvider(edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value", 1))) {
        ++count_;
      }
      static int count_;
    };
    int LoadableDummyProviderA::count_ = 0;

    class LoadableDummyESSourceA : public edm::eventsetup::test::DummyProxyProvider, public DummyFinder {
    public:
      LoadableDummyESSourceA(const edm::ParameterSet& iPSet)
          : DummyProxyProvider(edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value", 2))) {
        setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 0, 0)), edm::IOVSyncValue::endOfTime()));
        ++count_;
      }
      static int count_;
    };
    int LoadableDummyESSourceA::count_ = 0;
  }  // namespace other
  namespace cpu {
    class LoadableDummyFinderA : public DummyFinder {
    public:
      LoadableDummyFinderA(const edm::ParameterSet&) { ++count_; }
      static int count_;
    };
    int LoadableDummyFinderA::count_ = 0;

    class LoadableDummyProviderA : public edm::eventsetup::test::DummyProxyProvider {
    public:
      LoadableDummyProviderA(const edm::ParameterSet& iPSet)
          : DummyProxyProvider(edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value", 1))) {
        ++count_;
      }
      static int count_;
    };
    int LoadableDummyProviderA::count_ = 0;
    using LoadableDummyProviderB = LoadableDummyProviderA;

    class LoadableDummyESSourceA : public edm::eventsetup::test::DummyProxyProvider, public DummyFinder {
    public:
      LoadableDummyESSourceA(const edm::ParameterSet& iPSet)
          : DummyProxyProvider(edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value", 2))) {
        setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 0, 0)), edm::IOVSyncValue::endOfTime()));
        ++count_;
      }
      static int count_;
    };
    int LoadableDummyESSourceA::count_ = 0;
    using LoadableDummyESSourceB = LoadableDummyESSourceA;
  }  // namespace cpu
}  // namespace edm::test

DEFINE_FWK_EVENTSETUP_SOURCE(edm::test::other::LoadableDummyFinderA);
DEFINE_FWK_EVENTSETUP_SOURCE(edm::test::cpu::LoadableDummyFinderA);
DEFINE_FWK_EVENTSETUP_SOURCE(edm::test::other::LoadableDummyESSourceA);
DEFINE_FWK_EVENTSETUP_SOURCE(edm::test::cpu::LoadableDummyESSourceA);
DEFINE_FWK_EVENTSETUP_SOURCE(edm::test::cpu::LoadableDummyESSourceB);
DEFINE_FWK_EVENTSETUP_MODULE(edm::test::other::LoadableDummyProviderA);
DEFINE_FWK_EVENTSETUP_MODULE(edm::test::cpu::LoadableDummyProviderA);
DEFINE_FWK_EVENTSETUP_MODULE(edm::test::cpu::LoadableDummyProviderB);

class testEventsetupplugin : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEventsetupplugin);

  CPPUNIT_TEST(finderTest);
  CPPUNIT_TEST(simpleResolverTest);
  CPPUNIT_TEST(complexResolverTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void finderTest();
  void simpleResolverTest();
  void complexResolverTest();
  void configurableResolverTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetupplugin);

static void doInit() {
  static bool firstTime = true;
  if (firstTime) {
    if (not edmplugin::PluginManager::isAvailable()) {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    }
    firstTime = false;
  }
}

void testEventsetupplugin::finderTest()

{
  doInit();
  EventSetupsController esController;
  EventSetupProvider provider(&activityRegistry);
  edm::ModuleTypeResolverMaker const* resolverMaker = nullptr;

  edm::ParameterSet dummyFinderPSet;
  dummyFinderPSet.addParameter("@module_type", std::string("LoadableDummyFinder"));
  dummyFinderPSet.addParameter("@module_label", std::string(""));
  dummyFinderPSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummyFinderPSet, resolverMaker);

  ComponentDescription descFinder("LoadableDummyFinder", "", ComponentDescription::unknownID(), true);
  std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
  //should not be found since not a provider
  CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

  edm::ParameterSet dummyProviderPSet;
  dummyProviderPSet.addParameter("@module_type", std::string("LoadableDummyProvider"));
  dummyProviderPSet.addParameter("@module_label", std::string(""));
  dummyProviderPSet.registerIt();
  ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, resolverMaker);

  ComponentDescription desc("LoadableDummyProvider", "", ComponentDescription::unknownID(), false);
  descriptions = provider.proxyProviderDescriptions();
  CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
  CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);

  edm::ParameterSet dummySourcePSet;
  dummySourcePSet.addParameter("@module_type", std::string("LoadableDummyESSource"));
  dummySourcePSet.addParameter("@module_label", std::string(""));
  dummySourcePSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummySourcePSet, resolverMaker);

  ComponentDescription descSource("LoadableDummyESSource", "", ComponentDescription::unknownID(), true);
  descriptions = provider.proxyProviderDescriptions();
  CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
  CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
}

void testEventsetupplugin::simpleResolverTest() {
  doInit();

  edm::test::SimpleTestTypeResolverMaker resolverMaker;
  EventSetupsController esController(&resolverMaker);
  EventSetupProvider provider(&activityRegistry);

  edm::ParameterSet dummyFinderPSet;
  dummyFinderPSet.addParameter("@module_type", std::string("LoadableDummyFinder"));
  dummyFinderPSet.addParameter("@module_label", std::string(""));
  dummyFinderPSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummyFinderPSet, &resolverMaker);

  ComponentDescription descFinder("LoadableDummyFinder", "", ComponentDescription::unknownID(), true);
  std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
  //should not be found since not a provider
  CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

  edm::ParameterSet dummyProviderPSet;
  dummyProviderPSet.addParameter("@module_type", std::string("LoadableDummyProvider"));
  dummyProviderPSet.addParameter("@module_label", std::string(""));
  dummyProviderPSet.registerIt();
  ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

  ComponentDescription desc("LoadableDummyProvider", "", ComponentDescription::unknownID(), false);
  descriptions = provider.proxyProviderDescriptions();
  CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
  CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);

  edm::ParameterSet dummySourcePSet;
  dummySourcePSet.addParameter("@module_type", std::string("LoadableDummyESSource"));
  dummySourcePSet.addParameter("@module_label", std::string(""));
  dummySourcePSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummySourcePSet, &resolverMaker);

  ComponentDescription descSource("LoadableDummyESSource", "", ComponentDescription::unknownID(), true);
  descriptions = provider.proxyProviderDescriptions();
  CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
  CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
}

void testEventsetupplugin::complexResolverTest() {
  doInit();

  edm::test::ComplexTestTypeResolverMaker resolverMaker;
  EventSetupsController esController(&resolverMaker);
  EventSetupProvider provider(&activityRegistry);

  edm::ParameterSet dummyFinderPSet;
  dummyFinderPSet.addParameter("@module_type", std::string("generic::LoadableDummyFinderA"));
  dummyFinderPSet.addParameter("@module_label", std::string(""));
  dummyFinderPSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummyFinderPSet, &resolverMaker);

  ComponentDescription descFinder("generic::LoadableDummyFinderA", "", ComponentDescription::unknownID(), true);
  std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
  //should not be found since not a provider
  CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

  // find other
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 0);
    edm::ParameterSet dummyProviderPSet;
    dummyProviderPSet.addParameter("@module_type", std::string("generic::LoadableDummyProviderA"));
    dummyProviderPSet.addParameter("@module_label", std::string(""));
    dummyProviderPSet.registerIt();
    ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

    ComponentDescription desc("generic::LoadableDummyProviderA", "", ComponentDescription::unknownID(), false);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 1);
    edm::test::other::LoadableDummyProviderA::count_ = 0;
  }

  // find cpu
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderB::count_ == 0);
    edm::ParameterSet dummyProviderPSet;
    dummyProviderPSet.addParameter("@module_type", std::string("generic::LoadableDummyProviderB"));
    dummyProviderPSet.addParameter("@module_label", std::string(""));
    dummyProviderPSet.registerIt();
    ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

    ComponentDescription desc("generic::LoadableDummyProviderB", "", ComponentDescription::unknownID(), false);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderB::count_ == 1);
    edm::test::cpu::LoadableDummyProviderB::count_ = 0;
  }

  // find other
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyESSourceA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyESSourceA::count_ == 0);
    edm::ParameterSet dummySourcePSet;
    dummySourcePSet.addParameter("@module_type", std::string("generic::LoadableDummyESSourceA"));
    dummySourcePSet.addParameter("@module_label", std::string(""));
    dummySourcePSet.registerIt();
    SourceFactory::get()->addTo(esController, provider, dummySourcePSet, &resolverMaker);

    ComponentDescription descSource("generic::LoadableDummyESSourceA", "", ComponentDescription::unknownID(), true);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyESSourceA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyESSourceA::count_ == 1);
    edm::test::other::LoadableDummyESSourceA::count_ = 0;
  }

  // find cpu
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyESSourceB::count_ == 0);
    edm::ParameterSet dummySourcePSet;
    dummySourcePSet.addParameter("@module_type", std::string("generic::LoadableDummyESSourceB"));
    dummySourcePSet.addParameter("@module_label", std::string(""));
    dummySourcePSet.registerIt();
    SourceFactory::get()->addTo(esController, provider, dummySourcePSet, &resolverMaker);

    ComponentDescription descSource("generic::LoadableDummyESSourceB", "", ComponentDescription::unknownID(), true);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(descSource) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(descSource)) == descSource);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyESSourceB::count_ == 1);
    edm::test::cpu::LoadableDummyESSourceB::count_ = 0;
  }
}

void testEventsetupplugin::configurableResolverTest() {
  doInit();

  edm::test::ConfigurableTestTypeResolverMaker resolverMaker;
  EventSetupsController esController(&resolverMaker);
  EventSetupProvider provider(&activityRegistry);

  edm::ParameterSet dummyFinderPSet;
  dummyFinderPSet.addParameter("@module_type", std::string("generic::LoadableDummyFinderA"));
  dummyFinderPSet.addParameter("@module_label", std::string(""));
  dummyFinderPSet.addUntrackedParameter("variant", std::string(""));
  dummyFinderPSet.registerIt();
  SourceFactory::get()->addTo(esController, provider, dummyFinderPSet, &resolverMaker);

  ComponentDescription descFinder("generic::LoadableDummyFinderA", "", ComponentDescription::unknownID(), true);
  std::set<ComponentDescription> descriptions(provider.proxyProviderDescriptions());
  //should not be found since not a provider
  CPPUNIT_ASSERT(descriptions.find(descFinder) == descriptions.end());

  // default behavior
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 0);
    edm::ParameterSet dummyProviderPSet;
    dummyProviderPSet.addParameter("@module_type", std::string("generic::LoadableDummyProviderA"));
    dummyProviderPSet.addParameter("@module_label", std::string(""));
    dummyProviderPSet.addUntrackedParameter("variant", std::string(""));
    dummyProviderPSet.registerIt();
    ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

    ComponentDescription desc("generic::LoadableDummyProviderA", "", ComponentDescription::unknownID(), false);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 1);
    edm::test::other::LoadableDummyProviderA::count_ = 0;
  }

  // set variant to cpu
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 0);
    edm::ParameterSet dummyProviderPSet;
    dummyProviderPSet.addParameter("@module_type", std::string("generic::LoadableDummyProviderA"));
    dummyProviderPSet.addParameter("@module_label", std::string(""));
    dummyProviderPSet.addUntrackedParameter("variant", std::string("cpu"));
    dummyProviderPSet.registerIt();
    ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

    ComponentDescription desc("generic::LoadableDummyProviderA", "", ComponentDescription::unknownID(), false);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 1);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 0);
    edm::test::cpu::LoadableDummyProviderA::count_ = 0;
  }

  // set variant to other
  {
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 0);
    edm::ParameterSet dummyProviderPSet;
    dummyProviderPSet.addParameter("@module_type", std::string("generic::LoadableDummyProviderA"));
    dummyProviderPSet.addParameter("@module_label", std::string(""));
    dummyProviderPSet.addUntrackedParameter("variant", std::string("other"));
    dummyProviderPSet.registerIt();
    ModuleFactory::get()->addTo(esController, provider, dummyProviderPSet, &resolverMaker);

    ComponentDescription desc("generic::LoadableDummyProviderA", "", ComponentDescription::unknownID(), false);
    descriptions = provider.proxyProviderDescriptions();
    CPPUNIT_ASSERT(descriptions.find(desc) != descriptions.end());
    CPPUNIT_ASSERT(*(descriptions.find(desc)) == desc);
    CPPUNIT_ASSERT(edm::test::cpu::LoadableDummyProviderA::count_ == 0);
    CPPUNIT_ASSERT(edm::test::other::LoadableDummyProviderA::count_ == 1);
    edm::test::other::LoadableDummyProviderA::count_ = 0;
  }
}
