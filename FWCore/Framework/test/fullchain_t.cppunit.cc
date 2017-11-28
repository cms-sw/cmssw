/*
 *  full_chain_test.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/3/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */
#include <iostream>
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/HCMethods.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"
#include "cppunit/extensions/HelperMacros.h"

using namespace edm;
using namespace edm::eventsetup;
using namespace edm::eventsetup::test;

class testfullChain : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testfullChain);

  CPPUNIT_TEST(getfromDataproxyproviderTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void getfromDataproxyproviderTest();
};

/// registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfullChain);

void testfullChain::getfromDataproxyproviderTest() {
  eventsetup::EventSetupProvider provider;

  std::shared_ptr<DataProxyProvider> pProxyProv = std::make_shared<DummyProxyProvider>();
  provider.add(pProxyProv);

  std::shared_ptr<DummyFinder> pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  const Timestamp time_1(1);
  const IOVSyncValue sync_1(time_1);
  pFinder->setInterval(ValidityInterval(sync_1, IOVSyncValue(Timestamp(5))));
  for (unsigned int iTime = 1; iTime != 6; ++iTime) {
    const Timestamp time(iTime);
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(time));
    ESHandle<DummyData> pDummy;
    eventSetup.get<DummyRecord>().get(pDummy);
    CPPUNIT_ASSERT(0 != pDummy.product());

    eventSetup.getData(pDummy);

    CPPUNIT_ASSERT(0 != pDummy.product());
  }
}
