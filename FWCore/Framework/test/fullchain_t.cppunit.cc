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
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/src/SynchronousEventSetupsController.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "cppunit/extensions/HelperMacros.h"

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

  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer() : m_token{esConsumes()} {}

    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          rec->prefetchAsync(
              WaitingTaskHolder(group, &waitTask), proxies[i], &iImpl, edm::ServiceToken{}, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }

    ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> m_token;
  };
}  // namespace

class testfullChain : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testfullChain);

  CPPUNIT_TEST(getfromDataproxyproviderTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void getfromDataproxyproviderTest();

private:
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfullChain);

void testfullChain::getfromDataproxyproviderTest() {
  SynchronousEventSetupsController controller;
  ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  auto dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(1)), IOVSyncValue(Timestamp(5))));
  provider.add(dummyFinder);

  auto proxyProvider = std::make_shared<DummyProxyProvider>();
  provider.add(proxyProvider);

  edm::ESParentContext pc;
  for (unsigned int iTime = 1; iTime != 6; ++iTime) {
    const Timestamp time(iTime);
    controller.eventSetupForInstance(IOVSyncValue(time));
    DummyDataConsumer consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    EventSetup eventSetup(provider.eventSetupImpl(),
                          static_cast<unsigned int>(edm::Transition::Event),
                          consumer.esGetTokenIndices(edm::Transition::Event),
                          pc);
    ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());

    pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
  }
}
