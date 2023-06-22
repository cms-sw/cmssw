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
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/src/SynchronousEventSetupsController.h"
#include "FWCore/Framework/interface/es_Label.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
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
  CPPUNIT_TEST(getFromLambdaTest);
  CPPUNIT_TEST(getfromShareTest);
  CPPUNIT_TEST(getfromUniqueTest);
  CPPUNIT_TEST(getfromOptionalTest);
  CPPUNIT_TEST(decoratorTest);
  CPPUNIT_TEST(dependsOnTest);
  CPPUNIT_TEST(labelTest);
  CPPUNIT_TEST_EXCEPTION(failMultipleRegistration, cms::Exception);
  CPPUNIT_TEST(forceCacheClearTest);
  CPPUNIT_TEST(productResolverProviderTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void registerTest();
  void getFromTest();
  void getFromLambdaTest();
  void getfromShareTest();
  void getfromUniqueTest();
  void getfromOptionalTest();
  void decoratorTest();
  void dependsOnTest();
  void labelTest();
  void failMultipleRegistration();
  void forceCacheClearTest();
  void productResolverProviderTest();

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

  class LambdaProducer : public ESProducer {
  public:
    LambdaProducer() {
      setWhatProduced([data_ = DummyData()](const DummyRecord& /*iRecord*/) mutable {
        ++data_.value_;
        return std::shared_ptr<DummyData>(&data_, edm::do_nothing_deleter{});
      });
    }
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
  testProd.createKeyedResolvers(dummyRecordKey, 1);
  CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));
  CPPUNIT_ASSERT(!testProd.isUsingRecord(depRecordKey));
  const ESProductResolverProvider::KeyedResolvers& keyedResolvers = testProd.keyedResolvers(dummyRecordKey);

  CPPUNIT_ASSERT(keyedResolvers.size() == 1);
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
    consumer.updateLookup(provider.recordsToResolverIndices());
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

void testEsproducer::getFromLambdaTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  provider.add(std::make_shared<LambdaProducer>());

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToResolverIndices());
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

  std::shared_ptr<ESProductResolverProvider> pResolverProv = std::make_shared<ShareProducer>();
  provider.add(pResolverProv);

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToResolverIndices());
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

  std::shared_ptr<ESProductResolverProvider> pResolverProv = std::make_shared<UniqueProducer>();
  provider.add(pResolverProv);

  auto pFinder = std::make_shared<DummyFinder>();
  provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

  edm::ESParentContext parentC;
  for (int iTime = 1; iTime != 6; ++iTime) {
    const edm::Timestamp time(iTime);
    pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
    controller.eventSetupForInstance(edm::IOVSyncValue(time));
    DummyDataConsumer<DummyRecord> consumer;
    consumer.updateLookup(provider.recordsToResolverIndices());
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
    consumer.updateLookup(provider.recordsToResolverIndices());
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

    std::shared_ptr<ESProductResolverProvider> pResolverProv = std::make_shared<LabelledProducer>();
    provider.add(pResolverProv);

    auto pFinder = std::make_shared<DummyFinder>();
    provider.add(std::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

    edm::ESParentContext parentC;
    for (int iTime = 1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
      controller.eventSetupForInstance(edm::IOVSyncValue(time));
      DummyDataConsumer2 consumer(edm::ESInputTag("", "foo"), edm::ESInputTag("", "fi"), edm::ESInputTag("", "fum"));
      consumer.updateLookup(provider.recordsToResolverIndices());
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
    consumer.updateLookup(provider.recordsToResolverIndices());

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
    consumer.updateLookup(provider.recordsToResolverIndices());
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
    consumer.updateLookup(provider.recordsToResolverIndices());
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
    consumer.updateLookup(provider.recordsToResolverIndices());
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
  class TestESProductResolverProvider : public ESProductResolverProvider {
  public:
    TestESProductResolverProvider();

    class TestResolver : public ESProductResolver {
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
    std::shared_ptr<TestResolver> resolverDummy_0_0_;
    std::shared_ptr<TestResolver> resolverDummy_0_1_;
    std::shared_ptr<TestResolver> resolverDummy_0_2_;
    std::shared_ptr<TestResolver> resolverDummy2_0_0_;
    std::shared_ptr<TestResolver> resolverDummy2_1_0_;
    std::shared_ptr<TestResolver> resolverDummy2_2_0_;
    std::shared_ptr<TestResolver> resolverDep_0_0_;
    std::shared_ptr<TestResolver> resolverDep_0_1_;
    std::shared_ptr<TestResolver> resolverDep_1_0_;
    std::shared_ptr<TestResolver> resolverDep_1_1_;

  private:
    KeyedResolversVector registerProxies(const EventSetupRecordKey& recordKey, unsigned int iovIndex) override {
      KeyedResolversVector keyedResolversVector;
      if (recordKey == EventSetupRecordKey::makeKey<DummyRecord>()) {
        if (iovIndex == 0) {
          keyedResolversVector.emplace_back(dataKeyDummy_0_, resolverDummy_0_0_);
        } else if (iovIndex == 1) {
          keyedResolversVector.emplace_back(dataKeyDummy_0_, resolverDummy_0_1_);
        } else if (iovIndex == 2) {
          keyedResolversVector.emplace_back(dataKeyDummy_0_, resolverDummy_0_2_);
        }
      } else if (recordKey == EventSetupRecordKey::makeKey<Dummy2Record>()) {
        keyedResolversVector.emplace_back(dataKeyDummy2_0_, resolverDummy2_0_0_);
        keyedResolversVector.emplace_back(dataKeyDummy2_1_, resolverDummy2_1_0_);
        keyedResolversVector.emplace_back(dataKeyDummy2_2_, resolverDummy2_2_0_);
      } else if (recordKey == EventSetupRecordKey::makeKey<DepRecord>()) {
        if (iovIndex == 0) {
          keyedResolversVector.emplace_back(dataKeyDep_0_, resolverDep_0_0_);
          keyedResolversVector.emplace_back(dataKeyDep_1_, resolverDep_1_0_);
        } else if (iovIndex == 1) {
          keyedResolversVector.emplace_back(dataKeyDep_0_, resolverDep_0_1_);
          keyedResolversVector.emplace_back(dataKeyDep_1_, resolverDep_1_1_);
        }
      }
      return keyedResolversVector;
    }
  };

  TestESProductResolverProvider::TestESProductResolverProvider()
      : dataKeyDummy_0_(DataKey::makeTypeTag<DummyData>(), "Dummy_0"),
        dataKeyDummy2_0_(DataKey::makeTypeTag<DummyData>(), "Dummy2_0"),
        dataKeyDummy2_1_(DataKey::makeTypeTag<DummyData>(), "Dummy2_1"),
        dataKeyDummy2_2_(DataKey::makeTypeTag<DummyData>(), "Dummy2_2"),
        dataKeyDep_0_(DataKey::makeTypeTag<DummyData>(), "Dep_0"),
        dataKeyDep_1_(DataKey::makeTypeTag<DummyData>(), "Dep_1"),
        resolverDummy_0_0_(std::make_shared<TestResolver>()),
        resolverDummy_0_1_(std::make_shared<TestResolver>()),
        resolverDummy_0_2_(std::make_shared<TestResolver>()),
        resolverDummy2_0_0_(std::make_shared<TestResolver>()),
        resolverDummy2_1_0_(std::make_shared<TestResolver>()),
        resolverDummy2_2_0_(std::make_shared<TestResolver>()),
        resolverDep_0_0_(std::make_shared<TestResolver>()),
        resolverDep_0_1_(std::make_shared<TestResolver>()),
        resolverDep_1_0_(std::make_shared<TestResolver>()),
        resolverDep_1_1_(std::make_shared<TestResolver>()) {
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

void testEsproducer::productResolverProviderTest() {
  TestESProductResolverProvider productResolverProvider;
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  EventSetupRecordKey dummy2RecordKey = EventSetupRecordKey::makeKey<Dummy2Record>();
  EventSetupRecordKey depRecordKey = EventSetupRecordKey::makeKey<DepRecord>();
  EventSetupRecordKey depOn2RecordKey = EventSetupRecordKey::makeKey<DepOn2Record>();
  unsigned int nConcurrentIOVs = 3;
  productResolverProvider.createKeyedResolvers(dummyRecordKey, nConcurrentIOVs);
  CPPUNIT_ASSERT(productResolverProvider.isUsingRecord(dummyRecordKey));
  CPPUNIT_ASSERT(productResolverProvider.isUsingRecord(dummy2RecordKey));
  CPPUNIT_ASSERT(productResolverProvider.isUsingRecord(depRecordKey));
  CPPUNIT_ASSERT(productResolverProvider.isUsingRecord(depOn2RecordKey));

  std::set<EventSetupRecordKey> expectedKeys;
  expectedKeys.insert(dummyRecordKey);
  expectedKeys.insert(dummy2RecordKey);
  expectedKeys.insert(depRecordKey);
  expectedKeys.insert(depOn2RecordKey);

  auto keys = productResolverProvider.usingRecords();
  CPPUNIT_ASSERT(keys == expectedKeys);

  keys.clear();
  productResolverProvider.fillRecordsNotAllowingConcurrentIOVs(keys);
  expectedKeys.clear();
  expectedKeys.insert(dummy2RecordKey);
  CPPUNIT_ASSERT(keys == expectedKeys);

  nConcurrentIOVs = 1;
  productResolverProvider.createKeyedResolvers(dummy2RecordKey, nConcurrentIOVs);
  nConcurrentIOVs = 2;
  productResolverProvider.createKeyedResolvers(depRecordKey, nConcurrentIOVs);
  nConcurrentIOVs = 4;
  productResolverProvider.createKeyedResolvers(depOn2RecordKey, nConcurrentIOVs);

  ESProductResolverProvider::KeyedResolvers& keyedResolvers0 = productResolverProvider.keyedResolvers(dummyRecordKey, 0);
  CPPUNIT_ASSERT(keyedResolvers0.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedResolvers0.size() == 1);
  {
    auto it = keyedResolvers0.begin();
    auto itEnd = keyedResolvers0.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers1 = productResolverProvider.keyedResolvers(dummyRecordKey, 1);
  CPPUNIT_ASSERT(keyedResolvers1.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedResolvers1.size() == 1);
  {
    auto it = keyedResolvers1.begin();
    auto itEnd = keyedResolvers1.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy_0_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers2 = productResolverProvider.keyedResolvers(dummyRecordKey, 2);
  CPPUNIT_ASSERT(keyedResolvers2.recordKey() == dummyRecordKey);
  CPPUNIT_ASSERT(keyedResolvers2.size() == 1);
  {
    auto it = keyedResolvers2.begin();
    auto itEnd = keyedResolvers2.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy_0_2_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers3 = productResolverProvider.keyedResolvers(dummy2RecordKey, 0);
  CPPUNIT_ASSERT(keyedResolvers3.recordKey() == dummy2RecordKey);
  CPPUNIT_ASSERT(keyedResolvers3.size() == 3);
  {
    auto it = keyedResolvers3.begin();
    auto itEnd = keyedResolvers3.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy2_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy2_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy2_1_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy2_1_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDummy2_2_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDummy2_2_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers4 = productResolverProvider.keyedResolvers(depRecordKey, 0);
  CPPUNIT_ASSERT(keyedResolvers4.recordKey() == depRecordKey);
  CPPUNIT_ASSERT(keyedResolvers4.size() == 2);
  {
    auto it = keyedResolvers4.begin();
    auto itEnd = keyedResolvers4.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDep_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDep_0_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDep_1_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDep_1_0_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers5 = productResolverProvider.keyedResolvers(depRecordKey, 1);
  CPPUNIT_ASSERT(keyedResolvers5.recordKey() == depRecordKey);
  CPPUNIT_ASSERT(keyedResolvers5.size() == 2);
  {
    auto it = keyedResolvers5.begin();
    auto itEnd = keyedResolvers5.end();
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDep_0_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDep_0_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(it.dataKey() == productResolverProvider.dataKeyDep_1_);
    CPPUNIT_ASSERT(it.productResolver() == productResolverProvider.resolverDep_1_1_.get());
    CPPUNIT_ASSERT(it != itEnd);
    ++it;
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers6 = productResolverProvider.keyedResolvers(depOn2RecordKey, 0);
  CPPUNIT_ASSERT(keyedResolvers6.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedResolvers6.size() == 0);
  {
    auto it = keyedResolvers6.begin();
    auto itEnd = keyedResolvers6.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers7 = productResolverProvider.keyedResolvers(depOn2RecordKey, 1);
  CPPUNIT_ASSERT(keyedResolvers7.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedResolvers7.size() == 0);
  {
    auto it = keyedResolvers7.begin();
    auto itEnd = keyedResolvers7.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers8 = productResolverProvider.keyedResolvers(depOn2RecordKey, 2);
  CPPUNIT_ASSERT(keyedResolvers8.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedResolvers8.size() == 0);
  {
    auto it = keyedResolvers8.begin();
    auto itEnd = keyedResolvers8.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }
  ESProductResolverProvider::KeyedResolvers& keyedResolvers9 = productResolverProvider.keyedResolvers(depOn2RecordKey, 3);
  CPPUNIT_ASSERT(keyedResolvers9.recordKey() == depOn2RecordKey);
  CPPUNIT_ASSERT(keyedResolvers9.size() == 0);
  {
    auto it = keyedResolvers9.begin();
    auto itEnd = keyedResolvers9.end();
    CPPUNIT_ASSERT(!(it != itEnd));
  }

  CPPUNIT_ASSERT(keyedResolvers4.contains(productResolverProvider.dataKeyDep_0_));
  CPPUNIT_ASSERT(keyedResolvers4.contains(productResolverProvider.dataKeyDep_1_));
  CPPUNIT_ASSERT(!keyedResolvers4.contains(productResolverProvider.dataKeyDummy_0_));

  ESProductResolverProvider::KeyedResolvers keyedResolvers10(nullptr, 0);
  CPPUNIT_ASSERT(keyedResolvers10.unInitialized());
  CPPUNIT_ASSERT(!keyedResolvers0.unInitialized());
}
