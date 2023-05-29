/*
 *  eventsetup_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/24/05.
 *  Changed by Viji Sundararajan on 24-Jun-05.
 *
 */

// Note that repeatedly in this test we add the modules directly to
// the EventSetupProvider instead of having the controller use the
// plugin system and ModuleFactory to add the modules. This works OK
// in tests as long as there are no SubProcesses, otherwise this needs
// to be done as in a real job where the modules are added as plugins
// through the parameter set passed to the controller.

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

#include "FWCore/Framework/src/SynchronousEventSetupsController.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyEventSetupRecordRetriever.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"
#include "FWCore/Framework/test/DummyRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESProductTag.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "cppunit/extensions/HelperMacros.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace edm;
using namespace edm::eventsetup;
using edm::eventsetup::test::DummyData;
using edm::eventsetup::test::DummyProxyProvider;

namespace {

  bool non_null(const void* iPtr) { return iPtr != nullptr; }

  edm::ParameterSet createDummyPset() {
    edm::ParameterSet pset;
    std::vector<std::string> emptyVStrings;
    pset.addParameter<std::vector<std::string>>("@all_esprefers", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_essources", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_esmodules", emptyVStrings);
    return pset;
  }
}  // namespace

class testEventsetup : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEventsetup);

  CPPUNIT_TEST(constructTest);
  CPPUNIT_TEST(getTest);
  CPPUNIT_TEST(tryToGetTest);
  CPPUNIT_TEST_EXCEPTION(getExcTest, edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST(provenanceTest);
  CPPUNIT_TEST(getDataWithESGetTokenTest);
  CPPUNIT_TEST(getHandleWithESGetTokenTest);
  CPPUNIT_TEST(getTransientHandleWithESGetTokenTest);
  CPPUNIT_TEST(recordValidityTest);
  CPPUNIT_TEST_EXCEPTION(recordValidityExcTest, edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST(recordValidityNoFinderTest);
  CPPUNIT_TEST_EXCEPTION(recordValidityNoFinderExcTest, edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST(recordValidityProxyNoFinderTest);
  CPPUNIT_TEST_EXCEPTION(recordValidityProxyNoFinderExcTest, edm::eventsetup::NoRecordException<DummyRecord>);

  CPPUNIT_TEST_EXCEPTION(producerConflictTest, cms::Exception);
  CPPUNIT_TEST_EXCEPTION(sourceConflictTest, cms::Exception);
  CPPUNIT_TEST(twoSourceTest);
  CPPUNIT_TEST(sourceProducerResolutionTest);
  CPPUNIT_TEST(preferTest);

  CPPUNIT_TEST(introspectionTest);

  CPPUNIT_TEST(iovExtensionTest);
  CPPUNIT_TEST(resetProxiesTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void constructTest();
  void getTest();
  void tryToGetTest();
  void getExcTest();
  void recordValidityTest();
  void recordValidityExcTest();
  void recordValidityNoFinderTest();
  void recordValidityNoFinderExcTest();
  void recordValidityProxyNoFinderTest();
  void recordValidityProxyNoFinderExcTest();
  void provenanceTest();
  void getDataWithESGetTokenTest();
  void getHandleWithESGetTokenTest();
  void getTransientHandleWithESGetTokenTest();

  void producerConflictTest();
  void sourceConflictTest();
  void twoSourceTest();
  void sourceProducerResolutionTest();
  void preferTest();

  void introspectionTest();

  void iovExtensionTest();
  void resetProxiesTest();

private:
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;

  DummyData kGood{1};
  DummyData kBad{0};
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetup);

namespace {
  edm::ActivityRegistry activityRegistry;
}

void testEventsetup::constructTest() {
  eventsetup::EventSetupProvider provider(&activityRegistry);
  const Timestamp time(1);
  const IOVSyncValue timestamp(time);
  bool newEventSetupImpl = false;
  auto eventSetupImpl = provider.eventSetupForInstance(timestamp, newEventSetupImpl);
  CPPUNIT_ASSERT(non_null(eventSetupImpl.get()));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
}

// Note there is a similar test in dependentrecord_t.cppunit.cc
// named getTest() that tests get and tryToGet using an EventSetupProvider.
// No need to repeat that test here. The next two just test EventSetupImpl
// at the lowest level.

void testEventsetup::getTest() {
  EventSetupImpl eventSetupImpl;
  std::vector<eventsetup::EventSetupRecordKey> keys;
  EventSetupRecordKey key = EventSetupRecordKey::makeKey<DummyRecord>();
  keys.push_back(key);
  eventSetupImpl.setKeyIters(keys.begin(), keys.end());
  EventSetupRecordImpl dummyRecordImpl{key, &activityRegistry};
  eventSetupImpl.addRecordImpl(dummyRecordImpl);
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(eventSetupImpl, 0, nullptr, pc);
  const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
  CPPUNIT_ASSERT(&dummyRecordImpl == gottenRecord.impl_);
}

void testEventsetup::tryToGetTest() {
  EventSetupImpl eventSetupImpl;
  std::vector<eventsetup::EventSetupRecordKey> keys;
  EventSetupRecordKey key = EventSetupRecordKey::makeKey<DummyRecord>();
  keys.push_back(key);
  eventSetupImpl.setKeyIters(keys.begin(), keys.end());
  EventSetupRecordImpl dummyRecordImpl{key, &activityRegistry};
  eventSetupImpl.addRecordImpl(dummyRecordImpl);
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(eventSetupImpl, 0, nullptr, pc);
  std::optional<DummyRecord> gottenRecord = eventSetup.tryToGet<DummyRecord>();
  CPPUNIT_ASSERT(gottenRecord);
  CPPUNIT_ASSERT(&dummyRecordImpl == gottenRecord.value().impl_);
}

void testEventsetup::getExcTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
  controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  eventSetup.get<DummyRecord>();
}

class DummyFinder : public EventSetupRecordIntervalFinder {
public:
  DummyFinder() : EventSetupRecordIntervalFinder(), interval_() { this->findingRecord<DummyRecord>(); }

  void setInterval(const ValidityInterval& iInterval) { interval_ = iInterval; }

protected:
  virtual void setIntervalFor(const eventsetup::EventSetupRecordKey&,
                              const IOVSyncValue& iTime,
                              ValidityInterval& iInterval) {
    if (interval_.validFor(iTime)) {
      iInterval = interval_;
    } else {
      iInterval = ValidityInterval();
    }
  }

private:
  ValidityInterval interval_;
};

void testEventsetup::recordValidityTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

  // Note this manner of adding finders works OK in tests as long as there
  // are no SubProcesses, otherwise this needs to be done as in a real
  // job where they are added as plugins through the pset passed to the controller.
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, pc);
  CPPUNIT_ASSERT(!eventSetup1.tryToGet<DummyRecord>().has_value());

  const Timestamp time_2(2);
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(time_2), IOVSyncValue(Timestamp(3))));
  {
    controller.eventSetupForInstance(IOVSyncValue(time_2));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    eventSetup.get<DummyRecord>();
    CPPUNIT_ASSERT(eventSetup.tryToGet<DummyRecord>().has_value());
  }

  {
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(3)));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    eventSetup.get<DummyRecord>();
    CPPUNIT_ASSERT(eventSetup.tryToGet<DummyRecord>().has_value());
  }
  {
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(4)));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    CPPUNIT_ASSERT(!eventSetup.tryToGet<DummyRecord>().has_value());
  }
}

void testEventsetup::recordValidityExcTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

  // Note this manner of adding finders works OK in tests as long as there
  // are no SubProcesses, otherwise this needs to be done as in a real
  // job where they are added as plugins through the pset passed to the controller.
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  eventSetup.get<DummyRecord>();
}

void testEventsetup::recordValidityNoFinderTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  CPPUNIT_ASSERT(!eventSetup.tryToGet<DummyRecord>().has_value());
}

void testEventsetup::recordValidityNoFinderExcTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  eventSetup.get<DummyRecord>();
}

//create an instance of the register
static eventsetup::RecordDependencyRegister<DummyRecord> s_factory;

void testEventsetup::recordValidityProxyNoFinderTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<DummyProxyProvider>());

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  CPPUNIT_ASSERT(!eventSetup.tryToGet<DummyRecord>().has_value());
}

void testEventsetup::recordValidityProxyNoFinderExcTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<DummyProxyProvider>());

  Timestamp time_1(1);
  controller.eventSetupForInstance(IOVSyncValue(time_1));
  edm::ESParentContext pc;
  const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  eventSetup.get<DummyRecord>();
}

void testEventsetup::producerConflictTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, false);

  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  //checking for conflicts is delayed until first eventSetupForInstance
  controller.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
}

void testEventsetup::sourceConflictTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);

  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  //checking for conflicts is delayed until first eventSetupForInstance
  controller.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
}

void testEventsetup::twoSourceTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<edm::DummyEventSetupRecordRetriever>();
    std::shared_ptr<eventsetup::DataProxyProvider> providerPtr(dummyProv);
    std::shared_ptr<edm::EventSetupRecordIntervalFinder> finderPtr(dummyProv);
    edm::eventsetup::ComponentDescription description2("DummyEventSetupRecordRetriever", "", 0, true);
    dummyProv->setDescription(description2);
    provider.add(providerPtr);
    provider.add(finderPtr);
  }
  //checking for conflicts is delayed until first eventSetupForInstance
  edm::ESParentContext pc;
  controller.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  const edm::EventSetup eventSetup3(provider.eventSetupImpl(), 0, nullptr, pc);
  CPPUNIT_ASSERT(!eventSetup3.tryToGet<DummyRecord>().has_value());
  CPPUNIT_ASSERT(!eventSetup3.tryToGet<DummyEventSetupRecord>().has_value());
  controller.eventSetupForInstance(IOVSyncValue(Timestamp(3)));
  const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, pc);
  CPPUNIT_ASSERT(!eventSetup4.tryToGet<DummyRecord>().has_value());
  CPPUNIT_ASSERT(eventSetup4.tryToGet<DummyEventSetupRecord>().has_value());
  eventSetup4.get<DummyEventSetupRecord>();
}

namespace {
  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer(ESInputTag const& iTag) : m_token{esConsumes(iTag)} {}

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
    ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> m_tokenUninitialized;
  };
}  // namespace

void testEventsetup::provenanceTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testLabel", 1, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     pc);
    edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);

    CPPUNIT_ASSERT(kGood.value_ == data->value_);
    const edm::eventsetup::ComponentDescription* desc = data.description();
    CPPUNIT_ASSERT(desc->label_ == "testLabel");
  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

namespace {
  //This just tests that the constructs will properly compile
  class [[maybe_unused]] EDConsumesCollectorConsumer : public edm::EDConsumerBase {
    EDConsumesCollectorConsumer() {
      using edm::eventsetup::test::DummyData;
      {
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token1(
            consumesCollector().esConsumes<DummyData, edm::DefaultRecord>());
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token2(
            consumesCollector().esConsumes<DummyData, edm::DefaultRecord>(edm::ESInputTag("Blah")));
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token3(
            consumesCollector().esConsumes<DummyData, edm::DefaultRecord, edm::Transition::BeginRun>());
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token4(
            consumesCollector().esConsumes<DummyData, edm::DefaultRecord, edm::Transition::BeginRun>(
                edm::ESInputTag("Blah")));
      }
      {
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token1(consumesCollector().esConsumes());
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token2(
            consumesCollector().esConsumes(edm::ESInputTag("Blah")));
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token3(
            consumesCollector().esConsumes<edm::Transition::BeginRun>());
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token4(
            consumesCollector().esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("Blah")));
      }
      {
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token1;
        token1 = consumesCollector().esConsumes();
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token2;
        token2 = consumesCollector().esConsumes(edm::ESInputTag("Blah"));
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token3;
        token3 = consumesCollector().esConsumes<edm::Transition::BeginRun>();
        [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token4;
        token4 = consumesCollector().esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("Blah"));
      }

    }  // namespace
  };

  class ConsumesProducer : public ESProducer {
  public:
    ConsumesProducer() : token_{setWhatProduced(this, "consumes").consumes<edm::eventsetup::test::DummyData>()} {}
    std::unique_ptr<edm::eventsetup::test::DummyData> produce(const DummyRecord& iRecord) {
      auto const& data = iRecord.get(token_);
      return std::make_unique<edm::eventsetup::test::DummyData>(data);
    }

  private:
    edm::ESGetToken<edm::eventsetup::test::DummyData, DummyRecord> token_;
  };

  class ConsumesFromProducer : public ESProducer {
  public:
    ConsumesFromProducer()
        : token_{setWhatProduced(this, "consumesFrom").consumesFrom<edm::eventsetup::test::DummyData, DummyRecord>()} {}
    std::unique_ptr<edm::eventsetup::test::DummyData> produce(const DummyRecord& iRecord) {
      auto const& data = iRecord.get(token_);
      return std::make_unique<edm::eventsetup::test::DummyData>(data);
    }

  private:
    edm::ESGetToken<edm::eventsetup::test::DummyData, DummyRecord> token_;
  };

  //This is used only to test compilation
  class [[maybe_unused]] ESConsumesCollectorProducer : public ESProducer {
  public:
    struct Helper {
      Helper(ESConsumesCollector iCollector) {
        using edm::eventsetup::test::DummyData;
        {
          [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token1(
              iCollector.consumesFrom<DummyData, edm::DefaultRecord>());
          [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token2(
              iCollector.consumesFrom<DummyData, edm::DefaultRecord>(edm::ESInputTag("Blah")));
        }
        {
          [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token1(iCollector.consumes());
          [[maybe_unused]] edm::ESGetToken<DummyData, edm::DefaultRecord> token2(
              iCollector.consumes(edm::ESInputTag("Blah")));
        }
      }
    };

    ESConsumesCollectorProducer() : helper_(setWhatProduced(this, "consumesCollector")) {}

    std::unique_ptr<edm::eventsetup::test::DummyData> produce(const DummyRecord& iRecord) {
      return std::unique_ptr<edm::eventsetup::test::DummyData>();
    }

  private:
    Helper helper_;
  };

  class SetMayConsumeProducer : public ESProducer {
  public:
    SetMayConsumeProducer(bool iSucceed) : succeed_(iSucceed) {
      setWhatProduced(this, label(iSucceed))
          .setMayConsume(
              token_,
              [iSucceed](auto& get, edm::ESTransientHandle<edm::eventsetup::test::DummyData> const& handle) {
                if (iSucceed) {
                  return get("", "");
                }
                return get.nothing();
              },
              edm::ESProductTag<edm::eventsetup::test::DummyData, DummyRecord>("", ""));
    }
    std::unique_ptr<edm::eventsetup::test::DummyData> produce(const DummyRecord& iRecord) {
      auto const& data = iRecord.getHandle(token_);
      CPPUNIT_ASSERT(data.isValid() == succeed_);
      if (data.isValid()) {
        return std::make_unique<edm::eventsetup::test::DummyData>(*data);
      }
      return std::unique_ptr<edm::eventsetup::test::DummyData>();
    }

  private:
    static const char* label(bool iSucceed) noexcept {
      if (iSucceed) {
        return "setMayConsumeSucceed";
      }
      return "setMayConsumeFail";
    }

    edm::ESGetToken<edm::eventsetup::test::DummyData, DummyRecord> token_;
    bool succeed_;
  };

}  // namespace

void testEventsetup::getDataWithESGetTokenTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testTwo", 1, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("ConsumesProducer", "consumes", 2, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "consumes");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<ConsumesProducer>();
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("ConsumesFromProducer", "consumesFrom", 3, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "consumesFrom");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<ConsumesFromProducer>();
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("SetMayConsumeProducer", "setMayConsumeSuceed", 4, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "setMayConsumeSuceed");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<SetMayConsumeProducer>(true);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("SetMayConsumeProducer", "setMayConsumeFail", 5, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "setMayConsumeFail");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<SetMayConsumeProducer>(false);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }

    edm::ESParentContext pc;
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      auto const& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data.value_);
      bool uninitializedTokenThrewException = false;
      try {
        (void)eventSetup.getData(consumer.m_tokenUninitialized);
      } catch (cms::Exception& ex) {
        uninitializedTokenThrewException = true;
        CPPUNIT_ASSERT(ex.category() == "InvalidESGetToken");
      }
      CPPUNIT_ASSERT(uninitializedTokenThrewException);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      const DummyData& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      auto const& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data.value_);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      CPPUNIT_ASSERT_THROW(eventSetup.getData(consumer.m_token), cms::Exception);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "consumes")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      const DummyData& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "consumesFrom")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      const DummyData& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "setMayConsumeFail")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      CPPUNIT_ASSERT_THROW(eventSetup.getData(consumer.m_token), edm::eventsetup::MakeDataException);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "setMayConsumeSucceed")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      const DummyData& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }

  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testEventsetup::getHandleWithESGetTokenTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  edm::ESParentContext pc;
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testTwo", 1, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());

      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      CPPUNIT_ASSERT(not eventSetup.getHandle(consumer.m_token));
      CPPUNIT_ASSERT_THROW(*eventSetup.getHandle(consumer.m_token), cms::Exception);
    }

  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testEventsetup::getTransientHandleWithESGetTokenTest() {
  using edm::eventsetup::test::DummyData;
  using edm::eventsetup::test::DummyProxyProvider;
  DummyData kGood{1};
  DummyData kBad{0};

  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  edm::ESParentContext pc;
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "testTwo", 1, false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       pc};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       pc};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       pc};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       pc};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(not data);
      CPPUNIT_ASSERT_THROW(*data, cms::Exception);
    }

  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testEventsetup::sourceProducerResolutionTest() {
  {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    edm::ESParentContext pc;
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 1, false);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const EventSetup eventSetup{provider.eventSetupImpl(),
                                static_cast<unsigned int>(edm::Transition::Event),
                                consumer.esGetTokenIndices(edm::Transition::Event),
                                pc};
    edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(kGood.value_ == data->value_);
  }

  //reverse order
  {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, false);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 1, true);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    ESParentContext pc;
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    EventSetup eventSetup{provider.eventSetupImpl(),
                          static_cast<unsigned int>(edm::Transition::Event),
                          consumer.esGetTokenIndices(edm::Transition::Event),
                          pc};
    edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(kGood.value_ == data->value_);
  }
}

void testEventsetup::preferTest() {
  edm::ParameterSet pset = createDummyPset();

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));

  edm::ESParentContext pc;
  try {
    {
      SynchronousEventSetupsController controller;
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
      provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      //default means use all proxies
      preferInfo[ComponentDescription("DummyProxyProvider", "", ComponentDescription::unknownID(), false)] =
          recordToData;
      provider.setPreferredProviderInfo(preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "bad", 0, false);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 1, false);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
    }

    //sources
    {
      SynchronousEventSetupsController controller;
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
      provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      //default means use all proxies
      preferInfo[ComponentDescription("DummyProxyProvider", "", ComponentDescription::unknownID(), false)] =
          recordToData;
      provider.setPreferredProviderInfo(preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "bad", 1, true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
    }

    //specific name
    {
      SynchronousEventSetupsController controller;
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
      provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      recordToData.insert(
          std::make_pair(std::string("DummyRecord"), std::make_pair(std::string("DummyData"), std::string())));
      preferInfo[ComponentDescription("DummyProxyProvider", "", ComponentDescription::unknownID(), false)] =
          recordToData;
      provider.setPreferredProviderInfo(preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider", "bad", 1, true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
    }

  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testEventsetup::introspectionTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  edm::ESParentContext pc;
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider", "", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
    {
      EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};

      CPPUNIT_ASSERT(eventSetup.recordIsProvidedByAModule(dummyRecordKey));
      std::vector<edm::eventsetup::EventSetupRecordKey> recordKeys;
      eventSetup.fillAvailableRecordKeys(recordKeys);
      CPPUNIT_ASSERT(1 == recordKeys.size());
      CPPUNIT_ASSERT(dummyRecordKey == recordKeys[0]);
      auto record = eventSetup.find(recordKeys[0]);
      CPPUNIT_ASSERT(record.has_value());
    }
    // Intentionally an out of range sync value so the IOV is invalid
    // to test the find function with a record that exists in the
    // EventSetupImpl but has a null pointer.
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(4)));
    {
      EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};

      CPPUNIT_ASSERT(eventSetup.recordIsProvidedByAModule(dummyRecordKey));
      std::vector<edm::eventsetup::EventSetupRecordKey> recordKeys;
      eventSetup.fillAvailableRecordKeys(recordKeys);
      CPPUNIT_ASSERT(0 == recordKeys.size());
      auto record = eventSetup.find(dummyRecordKey);
      CPPUNIT_ASSERT(!record.has_value());

      // Just to try all cases test find with a record type not in the EventSetupImpl
      // at all.
      EventSetupRecordKey dummyRecordKey1 = EventSetupRecordKey::makeKey<DummyEventSetupRecord>();
      auto record1 = eventSetup.find(dummyRecordKey1);
      CPPUNIT_ASSERT(!record1.has_value());
    }
  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testEventsetup::iovExtensionTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> finder = std::make_shared<DummyFinder>();
  finder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(finder));

  edm::ESParentContext pc;
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(2)});
    EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
    CPPUNIT_ASSERT(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
  }
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(3)});
    EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
    eventSetup.get<DummyRecord>();
    CPPUNIT_ASSERT(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
  }
  //extending the IOV should not cause the cache to be reset
  finder->setInterval(ValidityInterval(IOVSyncValue{Timestamp{2}}, IOVSyncValue{Timestamp{4}}));
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(4)});
    EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
    CPPUNIT_ASSERT(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
  }

  //this is a new IOV so should get cache reset
  finder->setInterval(ValidityInterval(IOVSyncValue{Timestamp{5}}, IOVSyncValue{Timestamp{6}}));
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(5)});
    EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
    CPPUNIT_ASSERT(3 == eventSetup.get<DummyRecord>().cacheIdentifier());
  }
}

void testEventsetup::resetProxiesTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DummyFinder> finder = std::make_shared<DummyFinder>();
  finder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(finder));

  ComponentDescription description("DummyProxyProvider", "", 0, true);
  ParameterSet ps;
  ps.addParameter<std::string>("name", "test11");
  ps.registerIt();
  description.pid_ = ps.id();
  DummyData kOne{1};
  auto dummyProv = std::make_shared<DummyProxyProvider>(kOne);
  dummyProv->setDescription(description);
  provider.add(dummyProv);

  edm::ESParentContext pc;
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(2)});
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    EventSetup eventSetup{provider.eventSetupImpl(),
                          static_cast<unsigned int>(edm::Transition::Event),
                          consumer.esGetTokenIndices(edm::Transition::Event),
                          pc};
    CPPUNIT_ASSERT(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
    edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(data->value_ == 1);
  }
  provider.forceCacheClear();
  {
    controller.eventSetupForInstance(IOVSyncValue{Timestamp(2)});
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    EventSetup eventSetup{provider.eventSetupImpl(),
                          static_cast<unsigned int>(edm::Transition::Event),
                          consumer.esGetTokenIndices(edm::Transition::Event),
                          pc};
    eventSetup.get<DummyRecord>();
    CPPUNIT_ASSERT(3 == eventSetup.get<DummyRecord>().cacheIdentifier());
    dummyProv->incrementData();
    edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(data->value_ == 2);
  }
}
