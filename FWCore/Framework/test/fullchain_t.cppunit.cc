/*
 *  full_chain_test.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/3/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */

#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "cppunit/extensions/HelperMacros.h"
#include "tbb/task_scheduler_init.h"

#include <memory>
#include <string>
#include <vector>

using namespace edm;
using namespace edm::eventsetup;
using namespace edm::eventsetup::test;

namespace {
  ActivityRegistry activityRegistry;

  ParameterSet createDummyPset() {
    ParameterSet pset;
    std::vector<std::string> emptyVStrings;
    pset.addParameter<std::vector<std::string>>("@all_esprefers", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_essources", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_esmodules", emptyVStrings);
    return pset;
  }
}  // namespace

class testfullChain : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testfullChain);

  CPPUNIT_TEST(getfromDataproxyproviderTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<tbb::task_scheduler_init>(1); }
  void tearDown() {}

  void getfromDataproxyproviderTest();

private:
  edm::propagate_const<std::unique_ptr<tbb::task_scheduler_init>> m_scheduler;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfullChain);

void testfullChain::getfromDataproxyproviderTest() {
  EventSetupsController controller;
  ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  auto dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(1)), IOVSyncValue(Timestamp(5))));
  provider.add(dummyFinder);

  auto proxyProvider = std::make_shared<DummyProxyProvider>();
  provider.add(proxyProvider);

  for (unsigned int iTime = 1; iTime != 6; ++iTime) {
    const Timestamp time(iTime);
    controller.eventSetupForInstance(IOVSyncValue(time));
    EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, false);
    ESHandle<DummyData> pDummy;
    eventSetup.get<DummyRecord>().get(pDummy);
    CPPUNIT_ASSERT(0 != pDummy.product());

    eventSetup.getData(pDummy);
    CPPUNIT_ASSERT(0 != pDummy.product());
  }
}
