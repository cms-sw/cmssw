/*
 *  eventsetuprecord_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/29/05.
 *  Changed by Viji on 06/07/2005
 */

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "FWCore/Framework/interface/HCTypeTag.h"

#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESValidHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include <memory>
#include "oneapi/tbb/task_arena.h"

namespace {
  edm::ActivityRegistry activityRegistry;
}

using namespace edm;
using namespace edm::eventsetup;
namespace eventsetuprecord_t {
  class DummyRecord : public edm::eventsetup::EventSetupRecordImplementation<DummyRecord> {
  public:
  };
}  // namespace eventsetuprecord_t
//HCMethods<T, T, EventSetup, EventSetupRecordKey, EventSetupRecordKey::IdTag >
HCTYPETAG_HELPER_METHODS(eventsetuprecord_t::DummyRecord)

//create an instance of the factory
static eventsetup::RecordDependencyRegister<eventsetuprecord_t::DummyRecord> const s_factory;

namespace eventsetuprecord_t {
  class Dummy {};
}  // namespace eventsetuprecord_t
using eventsetuprecord_t::Dummy;
using eventsetuprecord_t::DummyRecord;
typedef edm::eventsetup::MakeDataException ExceptionType;
typedef edm::eventsetup::NoDataException<Dummy> NoDataExceptionType;

class testEventsetupRecord : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEventsetupRecord);

  CPPUNIT_TEST(resolverTest);
  CPPUNIT_TEST(getHandleTest);
  CPPUNIT_TEST(getWithTokenTest);
  CPPUNIT_TEST(doGetTest);
  CPPUNIT_TEST(resolverResetTest);
  CPPUNIT_TEST(introspectionTest);
  CPPUNIT_TEST(transientTest);

  CPPUNIT_TEST_EXCEPTION(getNodataExpTest, NoDataExceptionType);
  CPPUNIT_TEST_EXCEPTION(doGetExepTest, ExceptionType);

  CPPUNIT_TEST_SUITE_END();

public:
  testEventsetupRecord();
  void setUp();
  void tearDown() {}

  void resolverTest();
  void getHandleTest();
  void getWithTokenTest();
  void doGetTest();
  void resolverResetTest();
  void introspectionTest();
  void transientTest();

  void getNodataExpTest();
  void doGetExepTest();

  EventSetupRecordKey dummyRecordKey_;
  oneapi::tbb::task_arena taskArena_;
  EventSetupImpl eventSetupImpl_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetupRecord);

HCTYPETAG_HELPER_METHODS(Dummy)

class FailingDummyResolver : public eventsetup::ESProductResolverTemplate<DummyRecord, Dummy> {
protected:
  const value_type* make(const record_type&, const DataKey&) final { return nullptr; }
  void const* getAfterPrefetchImpl() const final { return nullptr; }
};

class WorkingDummyResolver : public eventsetup::ESProductResolverTemplate<DummyRecord, Dummy> {
public:
  WorkingDummyResolver(const Dummy* iDummy) : data_(iDummy), invalidateCalled_(false), invalidateTransientCalled_(false) {}

  bool invalidateCalled() const { return invalidateCalled_; }

  bool invalidateTransientCalled() const { return invalidateTransientCalled_; }

  void set(Dummy* iDummy) { data_ = iDummy; }

protected:
  const value_type* make(const record_type&, const DataKey&) final {
    invalidateCalled_ = false;
    invalidateTransientCalled_ = false;
    return data_;
  }
  void invalidateCache() final {
    invalidateCalled_ = true;
    eventsetup::ESProductResolverTemplate<DummyRecord, Dummy>::invalidateCache();
  }

  void invalidateTransientCache() override {
    invalidateTransientCalled_ = true;
    //check default behavior
    eventsetup::ESProductResolverTemplate<DummyRecord, Dummy>::invalidateTransientCache();
  }
  void const* getAfterPrefetchImpl() const override { return data_; }

private:
  const Dummy* data_;
  bool invalidateCalled_;
  bool invalidateTransientCalled_;
};

testEventsetupRecord::testEventsetupRecord() : taskArena_(1), eventSetupImpl_() {}
void testEventsetupRecord::setUp() { dummyRecordKey_ = EventSetupRecordKey::makeKey<DummyRecord>(); }

class WorkingDummyProvider : public edm::eventsetup::ESProductResolverProvider {
public:
  WorkingDummyProvider(const edm::eventsetup::DataKey& iKey, std::shared_ptr<WorkingDummyResolver> iResolver)
      : m_key(iKey), m_resolver(iResolver) {
    usingRecord<DummyRecord>();
  }

protected:
  KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
    KeyedResolversVector keyedResolversVector;
    keyedResolversVector.emplace_back(m_key, m_resolver);
    return keyedResolversVector;
  }

private:
  edm::eventsetup::DataKey m_key;
  std::shared_ptr<WorkingDummyResolver> m_resolver;
};

void testEventsetupRecord::resolverTest() {
  eventsetup::EventSetupRecordImpl dummyRecord{dummyRecordKey_, &activityRegistry};

  FailingDummyResolver dummyResolver;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  CPPUNIT_ASSERT(nullptr == dummyRecord.find(dummyDataKey));

  dummyRecord.add(dummyDataKey, &dummyResolver);
  CPPUNIT_ASSERT(&dummyResolver == dummyRecord.find(dummyDataKey));

  const DataKey dummyFredDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "fred");
  CPPUNIT_ASSERT(nullptr == dummyRecord.find(dummyFredDataKey));
}

namespace {
  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer(ESInputTag const& iTag) : m_token{esConsumes(iTag)} {}

    void prefetch(eventsetup::EventSetupRecordImpl const& iRec) const {
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask waitTask{group};
        edm::ServiceToken token;
        iRec.prefetchAsync(WaitingTaskHolder(group, &waitTask), proxies[i], nullptr, token, edm::ESParentContext{});
        waitTask.wait();
      }
    }

    ESGetToken<Dummy, DummyRecord> m_token;
  };

  struct DummyDataConsumerGeneric : public EDConsumerBase {
    explicit DummyDataConsumerGeneric(DataKey const& iKey)
        : m_token{esConsumes<>(eventsetup::EventSetupRecordKey::makeKey<DummyRecord>(), iKey)} {}

    void prefetch(eventsetup::EventSetupRecordImpl const& iRec) const {
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask waitTask{group};
        edm::ServiceToken token;
        iRec.prefetchAsync(WaitingTaskHolder(group, &waitTask), proxies[i], nullptr, token, edm::ESParentContext{});
        waitTask.wait();
      }
    }

    ESGetTokenGeneric m_token;
  };

}  // namespace

namespace {
  template <typename CONSUMER>
  struct SetupRecordT {
    eventsetup::EventSetupRecordImpl dummyRecordImpl;
    edm::EventSetupImpl& eventSetupImpl_;
    CONSUMER& consumer;
    //we need the DataKeys to stick around since references are being kept to them
    std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::ESProductResolver*>> proxies;
    // same for ESParentContext
    ESParentContext pc_;

    SetupRecordT(CONSUMER& iConsumer,
                 EventSetupRecordKey const& iKey,
                 EventSetupImpl& iEventSetup,
                 ActivityRegistry* iRegistry,
                 std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::ESProductResolver*>> iProxies)
        : dummyRecordImpl(iKey, iRegistry),
          eventSetupImpl_(iEventSetup),
          consumer(iConsumer),
          proxies(std::move(iProxies)) {
      for (auto const& d : proxies) {
        dummyRecordImpl.add(d.first, d.second);
      }

      ESRecordsToProductResolverIndices resolverIndices({iKey});
      std::vector<DataKey> dataKeys;
      dummyRecordImpl.fillRegisteredDataKeys(dataKeys);

      (void)resolverIndices.dataKeysInRecord(0, iKey, dataKeys, dummyRecordImpl.componentsForRegisteredDataKeys());

      iConsumer.updateLookup(resolverIndices);
      iConsumer.prefetch(dummyRecordImpl);
    }

    DummyRecord makeRecord() {
      DummyRecord ret;
      ret.setImpl(&dummyRecordImpl, 0, consumer.esGetTokenIndices(edm::Transition::Event), &eventSetupImpl_, &pc_);
      return ret;
    }
  };
  using SetupRecord = SetupRecordT<DummyDataConsumer>;
  using SetupGenericRecord = SetupRecordT<DummyDataConsumerGeneric>;
}  // namespace

void testEventsetupRecord::getHandleTest() {
  FailingDummyResolver dummyResolver;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  ESHandle<Dummy> dummyPtr;
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};
    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(not dummyRecord.getHandle(consumer.m_token));
    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(not dummyPtr.isValid());
    CPPUNIT_ASSERT(not dummyPtr);
    CPPUNIT_ASSERT(dummyPtr.failedToGet());
    CPPUNIT_ASSERT_THROW(*dummyPtr, NoDataExceptionType);
    CPPUNIT_ASSERT_THROW(makeESValid(dummyPtr), cms::Exception);
  }

  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.getHandle(consumer.m_token), ExceptionType);
  }
  Dummy myDummy;
  WorkingDummyResolver workingResolver(&myDummy);
  ComponentDescription cd;
  cd.label_ = "";
  cd.type_ = "DummyProd";
  workingResolver.setProviderDescription(&cd);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working");
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(!dummyPtr.failedToGet());
    CPPUNIT_ASSERT(dummyPtr.isValid());
    CPPUNIT_ASSERT(dummyPtr);

    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(not dummyRecord.getHandle(consumer.m_token));
    CPPUNIT_ASSERT_THROW(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
  }
  //check if label is set
  cd.label_ = "foo";
  {
    DummyDataConsumer consumer{edm::ESInputTag("foo", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(not dummyRecord.getHandle(consumer.m_token));
    CPPUNIT_ASSERT_THROW(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
  }
}

void testEventsetupRecord::getWithTokenTest() {
  FailingDummyResolver dummyResolver;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), NoDataExceptionType);
  }

  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), ExceptionType);
  }
  Dummy myDummy;
  WorkingDummyResolver workingResolver(&myDummy);
  ComponentDescription cd;
  cd.label_ = "";
  cd.type_ = "DummyProd";
  workingResolver.setProviderDescription(&cd);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working");
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);

    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);
    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), cms::Exception);
  }
  //check if label is set
  cd.label_ = "foo";
  {
    DummyDataConsumer consumer{edm::ESInputTag("foo", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);
    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{consumer,
                   dummyRecordKey_,
                   eventSetupImpl_,
                   &activityRegistry,
                   {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), cms::Exception);
  }
}

void testEventsetupRecord::getNodataExpTest() {
  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  edm::ESConsumesInfo consumesInfo;
  edm::ESConsumesCollectorT<DummyRecord> cc(&consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
  auto token = cc.consumes<Dummy>();
  std::vector<edm::ESResolverIndex> getTokenIndices{eventsetup::ESRecordsToProductResolverIndices::missingResolverIndex()};

  EventSetupRecordImpl recImpl(DummyRecord::keyForClass(), &activityRegistry);
  DummyRecord dummyRecord;
  ESParentContext pc;
  dummyRecord.setImpl(&recImpl, 0, getTokenIndices.data(), &eventSetupImpl_, &pc);
  FailingDummyResolver dummyResolver;

  ESHandle<Dummy> dummyPtr = dummyRecord.getHandle(token);
  *dummyPtr;
}

void testEventsetupRecord::doGetTest() {
  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  {
    DummyDataConsumerGeneric consumer{dummyDataKey};

    SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(!dummyRecord.doGet(consumer.m_token));
  }

  FailingDummyResolver dummyResolver;

  {
    DummyDataConsumerGeneric consumer{dummyDataKey};

    SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.doGet(consumer.m_token), ExceptionType);
  }
  Dummy myDummy;
  WorkingDummyResolver workingResolver(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working");

  {
    DummyDataConsumerGeneric consumer{workingDataKey};

    SetupGenericRecord sr{consumer,
                          dummyRecordKey_,
                          eventSetupImpl_,
                          &activityRegistry,
                          {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT(dummyRecord.doGet(consumer.m_token));
  }
}

namespace {
  ComponentDescription const* find(std::vector<DataKey> const& iKeys,
                                   std::vector<ComponentDescription const*> const& iComp,
                                   DataKey const& iKey) {
    return iComp[std::lower_bound(iKeys.begin(), iKeys.end(), iKey) - iKeys.begin()];
  }
}  // namespace

void testEventsetupRecord::introspectionTest() {
  eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};
  FailingDummyResolver dummyResolver;

  ComponentDescription cd1;
  cd1.label_ = "foo1";
  cd1.type_ = "DummyProd1";
  cd1.isSource_ = false;
  cd1.isLooper_ = false;
  dummyResolver.setProviderDescription(&cd1);

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

  std::vector<edm::eventsetup::DataKey> keys;
  dummyRecordImpl.fillRegisteredDataKeys(keys);
  CPPUNIT_ASSERT(keys.empty());

  DummyRecord dummyRecord;
  ESParentContext pc;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, &eventSetupImpl_, &pc);

  std::vector<ComponentDescription const*> esproducers;
  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.empty());

  std::vector<DataKey> referencedDataKeys;
  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  auto referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.empty());
  CPPUNIT_ASSERT(referencedComponents.empty());

  dummyRecordImpl.add(dummyDataKey, &dummyResolver);

  dummyRecord.fillRegisteredDataKeys(keys);
  CPPUNIT_ASSERT(1 == keys.size());

  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.size() == 1);
  CPPUNIT_ASSERT(esproducers[0] == &cd1);

  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.size() == 1);
  CPPUNIT_ASSERT(referencedComponents.size() == 1);
  CPPUNIT_ASSERT(referencedComponents[0] == &cd1);
  CPPUNIT_ASSERT(find(referencedDataKeys, referencedComponents, dummyDataKey) == &cd1);

  Dummy myDummy;
  WorkingDummyResolver workingResolver(&myDummy);

  ComponentDescription cd2;
  cd2.label_ = "foo2";
  cd2.type_ = "DummyProd2";
  cd2.isSource_ = true;
  cd2.isLooper_ = false;
  workingResolver.setProviderDescription(&cd2);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working");

  dummyRecordImpl.add(workingDataKey, &workingResolver);

  dummyRecord.fillRegisteredDataKeys(keys);
  CPPUNIT_ASSERT(2 == keys.size());

  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.size() == 1);

  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.size() == 2);
  CPPUNIT_ASSERT(referencedComponents.size() == 2);
  CPPUNIT_ASSERT(find(referencedDataKeys, referencedComponents, workingDataKey) == &cd2);

  Dummy myDummy3;
  WorkingDummyResolver workingResolver3(&myDummy3);

  ComponentDescription cd3;
  cd3.label_ = "foo3";
  cd3.type_ = "DummyProd3";
  cd3.isSource_ = false;
  cd3.isLooper_ = true;
  workingResolver3.setProviderDescription(&cd3);

  const DataKey workingDataKey3(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working3");

  dummyRecordImpl.add(workingDataKey3, &workingResolver3);

  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.size() == 1);

  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.size() == 3);
  CPPUNIT_ASSERT(referencedComponents.size() == 3);
  CPPUNIT_ASSERT(find(referencedDataKeys, referencedComponents, workingDataKey3) == &cd3);

  Dummy myDummy4;
  WorkingDummyResolver workingResolver4(&myDummy4);

  ComponentDescription cd4;
  cd4.label_ = "foo4";
  cd4.type_ = "DummyProd4";
  cd4.isSource_ = false;
  cd4.isLooper_ = false;
  workingResolver4.setProviderDescription(&cd4);

  const DataKey workingDataKey4(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "working4");

  dummyRecordImpl.add(workingDataKey4, &workingResolver4);

  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.size() == 2);
  CPPUNIT_ASSERT(esproducers[1] == &cd4);

  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.size() == 4);
  CPPUNIT_ASSERT(referencedComponents.size() == 4);
  CPPUNIT_ASSERT(find(referencedDataKeys, referencedComponents, workingDataKey4) == &cd4);

  dummyRecordImpl.clearProxies();
  dummyRecord.fillRegisteredDataKeys(keys);
  CPPUNIT_ASSERT(0 == keys.size());
}

void testEventsetupRecord::doGetExepTest() {
  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");
  {
    DummyDataConsumerGeneric consumer{dummyDataKey};

    SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(!dummyRecord.doGet(consumer.m_token));
  }

  {
    FailingDummyResolver dummyResolver;

    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    DummyDataConsumerGeneric consumer{dummyDataKey};

    SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(dummyRecord.doGet(consumer.m_token));
  }
}

void testEventsetupRecord::resolverResetTest() {
  auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  Dummy myDummy;
  std::shared_ptr<WorkingDummyResolver> workingResolver = std::make_shared<WorkingDummyResolver>(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "");
  DummyDataConsumer consumer{edm::ESInputTag("", "")};
  SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{workingDataKey, workingResolver.get()}}};
  DummyRecord dummyRecord = sr.makeRecord();

  edm::ESConsumesInfo consumesInfo;
  edm::ESConsumesCollectorT<DummyRecord> cc(&consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
  auto token = cc.consumes<Dummy>();
  std::vector<edm::ESResolverIndex> getTokenIndices{edm::ESResolverIndex(0)};

  std::shared_ptr<WorkingDummyProvider> wdProv = std::make_shared<WorkingDummyProvider>(workingDataKey, workingResolver);
  CPPUNIT_ASSERT(nullptr != wdProv.get());
  if (wdProv.get() == nullptr)
    return;  // To silence Coverity
  wdProv->createKeyedResolvers(DummyRecord::keyForClass(), 1);
  dummyProvider->add(wdProv);

  //this causes the proxies to actually be placed in the Record
  edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
  dummyProvider->usePreferred(pref);

  edm::ESHandle<Dummy> hDummy = dummyRecord.getHandle(token);

  CPPUNIT_ASSERT(&myDummy == &(*hDummy));

  Dummy myDummy2;
  workingResolver->set(&myDummy2);

  //should not change
  hDummy = dummyRecord.getHandle(token);
  CPPUNIT_ASSERT(&myDummy == &(*hDummy));
  CPPUNIT_ASSERT(!workingResolver->invalidateCalled());
  CPPUNIT_ASSERT(!workingResolver->invalidateTransientCalled());

  dummyProvider->resetProxies();
  CPPUNIT_ASSERT(workingResolver->invalidateCalled());
  CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled());
  consumer.prefetch(sr.dummyRecordImpl);
  hDummy = dummyRecord.getHandle(token);
  CPPUNIT_ASSERT(&myDummy2 == &(*hDummy));
  CPPUNIT_ASSERT(!workingResolver->invalidateCalled());
  CPPUNIT_ASSERT(!workingResolver->invalidateTransientCalled());
}

void testEventsetupRecord::transientTest() {
  // NEEDS TO BE REWRITTEN WHEN WE FIX OR REMOVE THE TRANSIENT
  // PARTS OF THE EVENTSETUP CODE WHICH IS CURRENTLY DISABLED

  auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  Dummy myDummy;
  std::shared_ptr<WorkingDummyResolver> workingResolver = std::make_shared<WorkingDummyResolver>(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "");
  DummyDataConsumer consumer{edm::ESInputTag("", "")};
  SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{workingDataKey, workingResolver.get()}}};
  DummyRecord dummyRecordNoConst = sr.makeRecord();
  EventSetupRecord const& dummyRecord = dummyRecordNoConst;

  edm::ESConsumesInfo consumesInfo;
  edm::ESConsumesCollectorT<DummyRecord> cc(&consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
  auto token = cc.consumes<Dummy>();
  std::vector<edm::ESResolverIndex> getTokenIndices{edm::ESResolverIndex(0)};

  eventsetup::EventSetupRecordImpl& nonConstDummyRecordImpl = *const_cast<EventSetupRecordImpl*>(dummyRecord.impl_);

  std::shared_ptr<WorkingDummyProvider> wdProv = std::make_shared<WorkingDummyProvider>(workingDataKey, workingResolver);
  wdProv->createKeyedResolvers(DummyRecord::keyForClass(), 1);
  dummyProvider->add(wdProv);

  //this causes the proxies to actually be placed in the Record
  edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
  dummyProvider->usePreferred(pref);

  //do a transient access to see if it clears properly
  edm::ESTransientHandle<Dummy> hTDummy;
  CPPUNIT_ASSERT(hTDummy.transientAccessOnly);
  hTDummy = dummyRecord.getHandleImpl<edm::ESTransientHandle>(token);

  CPPUNIT_ASSERT(&myDummy == &(*hTDummy));
  CPPUNIT_ASSERT(workingResolver->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled() == false);

  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingResolver->invalidateCalled());
  CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled());

  Dummy myDummy2;
  workingResolver->set(&myDummy2);

  //do non-transient access to make sure nothing resets now
  consumer.prefetch(sr.dummyRecordImpl);
  edm::ESHandle<Dummy> hDummy = dummyRecord.getHandleImpl<edm::ESHandle>(token);

  hDummy = dummyRecord.getHandleImpl<edm::ESHandle>(token);
  CPPUNIT_ASSERT(&myDummy2 == &(*hDummy));
  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingResolver->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled() == false);

  //do another transient access which should not do a reset since we have a non-transient access outstanding
  consumer.prefetch(sr.dummyRecordImpl);
  hDummy = dummyRecord.getHandleImpl<edm::ESHandle>(token);
  hTDummy = dummyRecord.getHandleImpl<edm::ESTransientHandle>(token);

  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingResolver->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled() == false);

  //Ask for a transient then a non transient to be sure we don't have an ordering problem
  {
    dummyProvider->resetProxies();
    Dummy myDummy3;
    workingResolver->set(&myDummy3);

    consumer.prefetch(sr.dummyRecordImpl);
    hDummy = dummyRecord.getHandleImpl<edm::ESHandle>(token);
    hTDummy = dummyRecord.getHandleImpl<edm::ESTransientHandle>(token);

    CPPUNIT_ASSERT(&myDummy3 == &(*hDummy));
    CPPUNIT_ASSERT(&myDummy3 == &(*hTDummy));
    nonConstDummyRecordImpl.resetIfTransientInProxies();
    CPPUNIT_ASSERT(workingResolver->invalidateCalled() == false);
    CPPUNIT_ASSERT(workingResolver->invalidateTransientCalled() == false);
  }
}
