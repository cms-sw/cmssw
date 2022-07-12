/*
 *  esproducer_t.cppunit.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 28-Jun-05
 */
#include <iostream>
#include "cppunit/extensions/HelperMacros.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/Dummy2Record.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DepOn2Record.h"
#include "FWCore/Framework/test/DepRecord.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/src/SynchronousEventSetupsController.h"
#include "FWCore/Framework/interface/es_Label.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include <memory>
#include <optional>

using edm::eventsetup::test::DummyData;
using namespace edm::eventsetup;
using edm::ESProducer;
using edm::EventSetupRecordIntervalFinder;

namespace {
  edm::ParameterSet createDummyPset() {
    edm::ParameterSet pset;
    std::vector<std::string> emptyVStrings;
    pset.addParameter<std::vector<std::string>>("@all_esprefers", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_essources", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_esmodules", emptyVStrings);
    return pset;
  }
  edm::ActivityRegistry activityRegistry;

  struct DummyDataConsumerBase : public edm::EDConsumerBase {
    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          rec->prefetchAsync(
              edm::WaitingTaskHolder(group, &waitTask), proxies[i], &iImpl, edm::ServiceToken{}, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }
  };

  template <typename Record>
  struct DummyDataConsumer : public DummyDataConsumerBase {
    DummyDataConsumer() : m_token{esConsumes()} {}
    edm::ESGetToken<DummyData, Record> m_token;
  };

  struct DummyDataConsumer2 : public DummyDataConsumerBase {
    DummyDataConsumer2(edm::ESInputTag const& tag1, edm::ESInputTag const& tag2, edm::ESInputTag const& tag3)
        : m_token1{esConsumes(tag1)}, m_token2{esConsumes(tag2)}, m_token3{esConsumes(tag3)} {}
    edm::ESGetToken<DummyData, DummyRecord> m_token1;
    edm::ESGetToken<DummyData, DummyRecord> m_token2;
    edm::ESGetToken<DummyData, DummyRecord> m_token3;
  };
}  // namespace

class testEsproducer : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEsproducer);

  CPPUNIT_TEST(registerTest);
  CPPUNIT_TEST(getFromTest);
  CPPUNIT_TEST(getfromShareTest);
  CPPUNIT_TEST(getfromUniqueTest);
  CPPUNIT_TEST(getfromOptionalTest);
  CPPUNIT_TEST(decoratorTest);
  CPPUNIT_TEST(dependsOnTest);
  CPPUNIT_TEST(labelTest);
  CPPUNIT_TEST_EXCEPTION(failMultipleRegistration, cms::Exception);
  CPPUNIT_TEST(forceCacheClearTest);
  CPPUNIT_TEST(dataProxyProviderTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void registerTest();
  void getFromTest();
  void getfromShareTest();
  void getfromUniqueTest();
  void getfromOptionalTest();
  void decoratorTest();
  void dependsOnTest();
  void labelTest();
  void failMultipleRegistration();
  void forceCacheClearTest();
  void dataProxyProviderTest();

private:
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;

  class Test1Producer : public ESProducer {
  public:
    Test1Producer() { setWhatProduced(this); }
    std::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++data_.value_;
      return std::shared_ptr<DummyData>(&data_, edm::do_nothing_deleter{});
    }

  private:
    DummyData data_{0};
  };

  class OptionalProducer : public ESProducer {
  public:
    OptionalProducer() { setWhatProduced(this); }
    std::optional<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++data_.value_;
      return data_;
    }

  private:
    DummyData data_{0};
  };

  class MultiRegisterProducer : public ESProducer {
  public:
    MultiRegisterProducer() {
      setWhatProduced(this);
      setWhatProduced(this);
    }
    std::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      return std::shared_ptr<DummyData>(&data_, edm::do_nothing_deleter{});
    }

  private:
    DummyData data_{0};
  };

  class ShareProducer : public ESProducer {
  public:
    ShareProducer() { setWhatProduced(this); }
    std::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++ptr_->value_;
      return ptr_;
    }

  private:
    std::shared_ptr<DummyData> ptr_{std::make_shared<DummyData>(0)};
  };

  class UniqueProducer : public ESProducer {
  public:
    UniqueProducer() { setWhatProduced(this); }
    std::unique_ptr<DummyData> produce(const DummyRecord&) {
      ++data_.value_;
      return std::make_unique<DummyData>(data_);
    }

  private:
    DummyData data_;
  };

  class LabelledProducer : public ESProducer {
  public:
    enum { kFi, kFum };
    typedef edm::ESProducts<edm::es::L<DummyData, kFi>, edm::es::L<DummyData, kFum>> ReturnProducts;
    LabelledProducer() {
      setWhatProduced(this, &LabelledProducer::produceMore, edm::es::label("fi", kFi)("fum", kFum));
      setWhatProduced(this, "foo");
    }

    std::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++ptr_->value_;
      return ptr_;
    }

    ReturnProducts produceMore(const DummyRecord&) {
      using edm::es::L;
      using namespace edm;
      ++fi_->value_;

      L<DummyData, kFum> fum(std::make_shared<DummyData>());
      fum->value_ = fi_->value_;

      return edm::es::products(fum, es::l<kFi>(fi_));
    }

  private:
    std::shared_ptr<DummyData> ptr_{std::make_shared<DummyData>(0)};
    std::shared_ptr<DummyData> fi_{std::make_shared<DummyData>(0)};
  };
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEsproducer);

void testEsproducer::registerTest() {
  Test1Producer testProd;
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  EventSetupRecordKey depRecordKey = EventSetupRecordKey::makeKey<DepRecord>();
  testProd.createKeyedProxies(dummyRecordKey, 1);
  CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));
  CPPUNIT_ASSERT(!testProd.isUsingRecord(depRecordKey));
  const DataProxyProvider::KeyedProxies& keyedProxies = testProd.keyedProxies(dummyRecordKey);

  CPPUNIT_ASSERT(keyedProxies.size() == 1);
}

void testEsproducer::getFromTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  // This manner of adding directly to the EventSetupProvider should work OK in tests
  // unless there are multiple EventSetupProviders (the case with SubProcesses).
  // Then there would be addition work to do to get things setup properly for the
  // functions that check for module sharing between EventSetupProviders.
  provider.add(std::make_shared<Test1Producer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(iTime == pDummy->value_);
  }
}

void testEsproducer::getfromShareTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DataProxyProvider> pProxyProv = std::make_shared<ShareProducer>();
  provider.add(pProxyProv);

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(iTime == pDummy->value_);
  }
}

void testEsproducer::getfromUniqueTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<DataProxyProvider> pProxyProv = std::make_shared<UniqueProducer>();
  provider.add(pProxyProv);

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(iTime == pDummy->value_);
  }
}

void testEsproducer::getfromOptionalTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<OptionalProducer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(iTime == pDummy->value_);
  }
}

void testEsproducer::labelTest() {
  try {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DataProxyProvider> pProxyProv = std::make_shared<LabelledProducer>();
    provider.add(pProxyProv);

    auto pFinder = std::make_shared<DummyFinder>();
    provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

    edm::ESParentContext parentC;
    for (int iTime = 1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
      controller.eventSetupForInstance(edm::IOVSyncValue(time));
      DummyDataConsumer2 consumer(edm::ESInputTag("", "foo"), edm::ESInputTag("", "fi"), edm::ESInputTag("", "fum"));
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC);

      edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token1);
      CPPUNIT_ASSERT(0 != pDummy.product());
      CPPUNIT_ASSERT(iTime == pDummy->value_);

      pDummy = eventSetup.getHandle(consumer.m_token2);
      CPPUNIT_ASSERT(0 != pDummy.product());
      CPPUNIT_ASSERT(iTime == pDummy->value_);

      pDummy = eventSetup.getHandle(consumer.m_token3);
      CPPUNIT_ASSERT(0 != pDummy.product());
      CPPUNIT_ASSERT(iTime == pDummy->value_);
    }
  } catch (const cms::Exception& iException) {
    std::cout << "caught exception " << iException.explainSelf() << std::endl;
    throw;
  }
}

struct TestDecorator {
  static int s_pre;
  static int s_post;

  void pre(const DummyRecord&) { ++s_pre; }

  void post(const DummyRecord&) { ++s_post; }
};

int TestDecorator::s_pre = 0;
int TestDecorator::s_post = 0;

class DecoratorProducer : public ESProducer {
public:
  DecoratorProducer() { setWhatProduced(this, TestDecorator{}); }
  std::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
    ++ptr_->value_;
    return ptr_;
  }

private:
  std::shared_ptr<DummyData> ptr_{std::make_shared<DummyData>(0)};
};

void testEsproducer::decoratorTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<DecoratorProducer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());

    CPPUNIT_ASSERT(iTime - 1 == TestDecorator::s_pre);
    CPPUNIT_ASSERT(iTime - 1 == TestDecorator::s_post);
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(iTime == TestDecorator::s_pre);
    CPPUNIT_ASSERT(iTime == TestDecorator::s_post);
    CPPUNIT_ASSERT(iTime == pDummy->value_);
  }
}

class DepProducer : public ESProducer {
public:
  DepProducer() {
    setWhatProduced(this,
                    dependsOn(&DepProducer::callWhenDummyChanges,
                              &DepProducer::callWhenDummyChanges2,
                              &DepProducer::callWhenDummyChanges3));
  }
  std::shared_ptr<DummyData> produce(const DepRecord& /*iRecord*/) { return ptr_; }
  void callWhenDummyChanges(const DummyRecord&) { ++ptr_->value_; }
  void callWhenDummyChanges2(const DummyRecord&) { ++ptr_->value_; }
  void callWhenDummyChanges3(const DummyRecord&) { ++ptr_->value_; }

private:
  std::shared_ptr<DummyData> ptr_{std::make_shared<DummyData>(0)};
};

void testEsproducer::dependsOnTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<DepProducer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DepRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(3 * iTime == pDummy->value_);
  }
}

void testEsproducer::failMultipleRegistration() { MultiRegisterProducer dummy; }

void testEsproducer::forceCacheClearTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<Test1Producer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  const edm::Timestamp time(1);
  pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
  controller.eventSetupForInstance(edm::IOVSyncValue(time));
  {
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    edm::ESParentContext parentC;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                     static_cast<unsigned int>(edm::Transition::Event),
                                     consumer.esGetTokenIndices(edm::Transition::Event),
                                     parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(1 == pDummy->value_);
  }
  provider.forceCacheClear();
  controller.eventSetupForInstance(edm::IOVSyncValue(time));
  {
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToProxyIndices());
    consumer.prefetch(provider.eventSetupImpl());
    edm::ESParentContext parentC;
    const edm::EventSetup eventSetup2(provider.eventSetupImpl(),
                                      static_cast<unsigned int>(edm::Transition::Event),
                                      consumer.esGetTokenIndices(edm::Transition::Event),
                                      parentC);
    edm::ESHandle<DummyData> pDummy = eventSetup2.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(0 != pDummy.product());
    CPPUNIT_ASSERT(2 == pDummy->value_);
  }
}

namespace {
  class TestDataProxyProvider : public DataProxyProvider {
  public:
    TestDataProxyProvider();

    class TestProxy : public DataProxy {
    public:
      void prefetchAsyncImpl(edm::WaitingTaskHolder,
                             EventSetupRecordImpl const&,
                             DataKey const&,
                             edm::EventSetupImpl const*,
                             edm::ServiceToken const&,
                             edm::ESParentContext const&) override {}
      void invalidateCache() override {}
      void const* getAfterPrefetchImpl() const override { return nullptr; }
    };

    DataKey dataKeyDummy_0_;
    DataKey dataKeyDummy2_0_;
    DataKey dataKeyDummy2_1_;
    DataKey dataKeyDummy2_2_;
    DataKey dataKeyDep_0_;
    DataKey dataKeyDep_1_;
    std::shared_ptr<TestProxy> proxyDummy_0_0_;
    std::shared_ptr<TestProxy> proxyDummy_0_1_;
    std::shared_ptr<TestProxy> proxyDummy_0_2_;
    std::shared_ptr<TestProxy> proxyDummy2_0_0_;
    std::shared_ptr<TestProxy> proxyDummy2_1_0_;
    std::shared_ptr<TestProxy> proxyDummy2_2_0_;
    std::shared_ptr<TestProxy> proxyDep_0_0_;
    std::shared_ptr<TestProxy> proxyDep_0_1_;
    std::shared_ptr<TestProxy> proxyDep_1_0_;
    std::shared_ptr<TestProxy> proxyDep_1_1_;

  private:
    KeyedProxiesVector registerProxies(const EventSetupRecordKey& recordKey, unsigned int iovIndex) override {
      KeyedProxiesVector keyedProxiesVector;
      if (recordKey == EventSetupRecordKey::makeKey<DummyRecord>()) {
        if (iovIndex == 0) {
          keyedProxiesVector.emplace_back(dataKeyDummy_0_, proxyDummy_0_0_);
        } else if (iovIndex == 1) {
          keyedProxiesVector.emplace_back(dataKeyDummy_0_, proxyDummy_0_1_);
        } else if (iovIndex == 2) {
          keyedProxiesVector.emplace_back(dataKeyDummy_0_, proxyDummy_0_2_);
        }
      } else if (recordKey == EventSetupRecordKey::makeKey<Dummy2Record>()) {
        keyedProxiesVector.emplace_back(dataKeyDummy2_0_, proxyDummy2_0_0_);
        keyedProxiesVector.emplace_back(dataKeyDummy2_1_, proxyDummy2_1_0_);
        keyedProxiesVector.emplace_back(dataKeyDummy2_2_, proxyDummy2_2_0_);
      } else if (recordKey == EventSetupRecordKey::makeKey<DepRecord>()) {
        if (iovIndex == 0) {
          keyedProxiesVector.emplace_back(dataKeyDep_0_, proxyDep_0_0_);
          keyedProxiesVector.emplace_back(dataKeyDep_1_, proxyDep_1_0_);
        } else if (iovIndex == 1) {
          keyedProxiesVector.emplace_back(dataKeyDep_0_, proxyDep_0_1_);
          keyedProxiesVector.emplace_back(dataKeyDep_1_, proxyDep_1_1_);
        }
      }
      return keyedProxiesVector;
    }
  };

  TestDataProxyProvider::TestDataProxyProvider()
      : dataKeyDummy_0_(DataKey::makeTypeTag<DummyData>(), "Dummy_0"),
        dataKeyDummy2_0_(DataKey::makeTypeTag<DummyData>(), "Dummy2_0"),
        dataKeyDummy2_1_(DataKey::makeTypeTag<DummyData>(), "Dummy2_1"),
        dataKeyDummy2_2_(DataKey::makeTypeTag<DummyData>(), "Dummy2_2"),
        dataKeyDep_0_(DataKey::makeTypeTag<DummyData>(), "Dep_0"),
        dataKeyDep_1_(DataKey::makeTypeTag<DummyData>(), "Dep_1"),
        proxyDummy_0_0_(std::make_shared<TestProxy>()),
        proxyDummy_0_1_(std::make_shared<TestProxy>()),
        proxyDummy_0_2_(std::make_shared<TestProxy>()),
        proxyDummy2_0_0_(std::make_shared<TestProxy>()),
        proxyDummy2_1_0_(std::make_shared<TestProxy>()),
        proxyDummy2_2_0_(std::make_shared<TestProxy>()),
        proxyDep_0_0_(std::make_shared<TestProxy>()),
        proxyDep_0_1_(std::make_shared<TestProxy>()),
        proxyDep_1_0_(std::make_shared<TestProxy>()),
        proxyDep_1_1_(std::make_shared<TestProxy>()) {
    usingRecord<DummyRecord>();
    usingRecord<Dummy2Record>();
    usingRecord<DepRecord>();
    usingRecord<DepOn2Record>();
    usingRecord<DummyRecord>();
    usingRecord<Dummy2Record>();
    usingRecord<DepRecord>();
    usingRecord<DepOn2Record>();
  }
}  // namespace

void testEsproducer::dataProxyProviderTest() {
  TestDataProxyProvider dataProxyProvider;
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  EventSetupRecordKey dummy2RecordKey = EventSetupRecordKey::makeKey<Dummy2Record>();
  EventSetupRecordKey depRecordKey = EventSetupRecordKey::makeKey<DepRecord>();
  EventSetupRecordKey depOn2RecordKey = EventSetupRecordKey::makeKey<DepOn2Record>();
  unsigned int nConcurrentIOVs = 3;
  dataProxyProvider.createKeyedProxies(dummyRecordKey, nConcurrentIOVs);
  CPPUNIT_ASSERT(dataProxyProvider.isUsingRecord(dummyRecordKey));
  CPPUNIT_ASSERT(dataProxyProvider.isUsingRecord(dummy2RecordKey));
  CPPUNIT_ASSERT(dataProxyProvider.isUsingRecord(depRecordKey));
  CPPUNIT_ASSERT(dataProxyProvider.isUsingRecord(depOn2RecordKey));

  std::set<EventSetupRecordKey> expectedKeys;
  expectedKeys.insert(dummyRecordKey);
  expectedKeys.insert(dummy2RecordKey);
  expectedKeys.insert(depRecordKey);
  expectedKeys.insert(depOn2RecordKey);

  auto keys = dataProxyProvider.usingRecords();
  CPPUNIT_ASSERT(keys == expectedKeys);

  keys.clear();
  dataProxyProvider.fillRecordsNotAllowingConcurrentIOVs(keys);
  expectedKeys.clear();
  expectedKeys.insert(dummy2RecordKey);
  CPPUNIT_ASSERT(keys == expectedKeys);

  nConcurrentIOVs = 1;
  dataProxyProvider.createKeyedProxies(dummy2RecordKey, nConcurrentIOVs);
  nConcurrentIOVs = 2;
  dataProxyProvider.createKeyedProxies(depRecordKey, nConcurrentIOVs);
  nConcurrentIOVs = 4;
  dataProxyProvider.createKeyedProxies(depOn2RecordKey, nConcurrentIOVs);

  DataProxyProvider::KeyedProxies& keyedProxies0 = dataProxyProvider.keyedProxies(dummyRecordKey, 0);
  CPPUNIT_ASSERT(keyedProxies0.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedProxies0.size() == 1);
  {
    auto it = keyedProxies0.begin();
    auto itEnd = keyedProxies0.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies1 = dataProxyProvider.keyedProxies(dummyRecordKey, 1);
  CPPUNIT_ASSERT(keyedProxies1.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedProxies1.size() == 1);
  {
    auto it = keyedProxies1.begin();
    auto itEnd = keyedProxies1.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy_0_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies2 = dataProxyProvider.keyedProxies(dummyRecordKey, 2);
  CPPUNIT_ASSERT(keyedProxies2.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedProxies2.size() == 1);
  {
    auto it = keyedProxies2.begin();
    auto itEnd = keyedProxies2.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy_0_2_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies3 = dataProxyProvider.keyedProxies(dummy2RecordKey, 0);
  CPPUNIT_ASSERT(keyedProxies3.recordKey() == dummy2RecordKey);
  CPPUNIT_ASSERT(keyedProxies3.size() == 3);
  {
    auto it = keyedProxies3.begin();
    auto itEnd = keyedProxies3.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy2_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy2_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy2_1_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy2_1_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDummy2_2_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDummy2_2_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies4 = dataProxyProvider.keyedProxies(depRecordKey, 0);
  CPPUNIT_ASSERT(keyedProxies4.recordKey() == depRecordKey);
  CPPUNIT_ASSERT(keyedProxies4.size() == 2);
  {
    auto it = keyedProxies4.begin();
    auto itEnd = keyedProxies4.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDep_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDep_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDep_1_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDep_1_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies5 = dataProxyProvider.keyedProxies(depRecordKey, 1);
  CPPUNIT_ASSERT(keyedProxies5.recordKey() == depRecordKey);
  CPPUNIT_ASSERT(keyedProxies5.size() == 2);
  {
    auto it = keyedProxies5.begin();
    auto itEnd = keyedProxies5.end();
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDep_0_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDep_0_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == dataProxyProvider.dataKeyDep_1_);
    CPPUNIT_ASSERT(it.dataProxy() == dataProxyProvider.proxyDep_1_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies6 = dataProxyProvider.keyedProxies(depOn2RecordKey, 0);
  CPPUNIT_ASSERT(keyedProxies6.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedProxies6.size() == 0);
  {
    auto it = keyedProxies6.begin();
    auto itEnd = keyedProxies6.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies7 = dataProxyProvider.keyedProxies(depOn2RecordKey, 1);
  CPPUNIT_ASSERT(keyedProxies7.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedProxies7.size() == 0);
  {
    auto it = keyedProxies7.begin();
    auto itEnd = keyedProxies7.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies8 = dataProxyProvider.keyedProxies(depOn2RecordKey, 2);
  CPPUNIT_ASSERT(keyedProxies8.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedProxies8.size() == 0);
  {
    auto it = keyedProxies8.begin();
    auto itEnd = keyedProxies8.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  DataProxyProvider::KeyedProxies& keyedProxies9 = dataProxyProvider.keyedProxies(depOn2RecordKey, 3);
  CPPUNIT_ASSERT(keyedProxies9.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedProxies9.size() == 0);
  {
    auto it = keyedProxies9.begin();
    auto itEnd = keyedProxies9.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }

  CPPUNIT_ASSERT(keyedProxies4.contains(dataProxyProvider.dataKeyDep_0_));
  CPPUNIT_ASSERT(keyedProxies4.contains(dataProxyProvider.dataKeyDep_1_));
  CPPUNIT_ASSERT(!keyedProxies4.contains(dataProxyProvider.dataKeyDummy_0_));

  DataProxyProvider::KeyedProxies keyedProxies10(nullptr, 0);
  CPPUNIT_ASSERT(keyedProxies10.unInitialized());
  CPPUNIT_ASSERT(!keyedProxies0.unInitialized());
}
