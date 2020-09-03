/*
 *  eventsetuprecord_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/29/05.
 *  Changed by Viji on 06/07/2005
 */

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "FWCore/Framework/interface/HCTypeTag.h"

#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESValidHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <memory>

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

  CPPUNIT_TEST(proxyTest);
  CPPUNIT_TEST(getTest);
  CPPUNIT_TEST(getHandleTest);
  CPPUNIT_TEST(getWithTokenTest);
  CPPUNIT_TEST(doGetTest);
  CPPUNIT_TEST(proxyResetTest);
  CPPUNIT_TEST(introspectionTest);
  CPPUNIT_TEST(transientTest);

  CPPUNIT_TEST_EXCEPTION(getNodataExpTest, NoDataExceptionType);
  CPPUNIT_TEST_EXCEPTION(getExepTest, ExceptionType);
  CPPUNIT_TEST_EXCEPTION(doGetExepTest, ExceptionType);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown() {}

  void proxyTest();
  void getTest();
  void getHandleTest();
  void getWithTokenTest();
  void doGetTest();
  void proxyResetTest();
  void introspectionTest();
  void transientTest();

  void getNodataExpTest();
  void getExepTest();
  void doGetExepTest();

  EventSetupRecordKey dummyRecordKey_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetupRecord);

HCTYPETAG_HELPER_METHODS(Dummy)

class FailingDummyProxy : public eventsetup::DataProxyTemplate<DummyRecord, Dummy> {
protected:
  const value_type* make(const record_type&, const DataKey&) final { return nullptr; }
  void const* getAfterPrefetchImpl() const final { return nullptr; }
};

class WorkingDummyProxy : public eventsetup::DataProxyTemplate<DummyRecord, Dummy> {
public:
  WorkingDummyProxy(const Dummy* iDummy) : data_(iDummy), invalidateCalled_(false), invalidateTransientCalled_(false) {}

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
    eventsetup::DataProxyTemplate<DummyRecord, Dummy>::invalidateCache();
  }

  void invalidateTransientCache() override {
    invalidateTransientCalled_ = true;
    //check default behavior
    eventsetup::DataProxyTemplate<DummyRecord, Dummy>::invalidateTransientCache();
  }
  void const* getAfterPrefetchImpl() const override { return data_; }

private:
  const Dummy* data_;
  bool invalidateCalled_;
  bool invalidateTransientCalled_;
};

void testEventsetupRecord::setUp() { dummyRecordKey_ = EventSetupRecordKey::makeKey<DummyRecord>(); }

class WorkingDummyProvider : public edm::eventsetup::DataProxyProvider {
public:
  WorkingDummyProvider(const edm::eventsetup::DataKey& iKey, std::shared_ptr<WorkingDummyProxy> iProxy)
      : m_key(iKey), m_proxy(iProxy) {
    usingRecord<DummyRecord>();
  }

protected:
  KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
    KeyedProxiesVector keyedProxiesVector;
    keyedProxiesVector.emplace_back(m_key, m_proxy);
    return keyedProxiesVector;
  }

private:
  edm::eventsetup::DataKey m_key;
  std::shared_ptr<WorkingDummyProxy> m_proxy;
};

void testEventsetupRecord::proxyTest() {
  eventsetup::EventSetupRecordImpl dummyRecord{dummyRecordKey_, &activityRegistry};

  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  CPPUNIT_ASSERT(nullptr == dummyRecord.find(dummyDataKey));

  dummyRecord.add(dummyDataKey, &dummyProxy);
  CPPUNIT_ASSERT(&dummyProxy == dummyRecord.find(dummyDataKey));

  const DataKey dummyFredDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "fred");
  CPPUNIT_ASSERT(nullptr == dummyRecord.find(dummyFredDataKey));
}

void testEventsetupRecord::getTest() {
  eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};

  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, nullptr, false);
  ESHandle<Dummy> dummyPtr;
  CPPUNIT_ASSERT(not dummyRecord.get(dummyPtr));
  CPPUNIT_ASSERT(not dummyPtr.isValid());
  CPPUNIT_ASSERT(not dummyPtr);
  CPPUNIT_ASSERT(dummyPtr.failedToGet());
  CPPUNIT_ASSERT_THROW(*dummyPtr, NoDataExceptionType);
  CPPUNIT_ASSERT_THROW(makeESValid(dummyPtr), cms::Exception);
  //CDJ do this replace
  //CPPUNIT_ASSERT_THROW(dummyRecord.get(dummyPtr),NoDataExceptionType);

  dummyRecordImpl.add(dummyDataKey, &dummyProxy);

  //dummyRecord.get(dummyPtr);
  CPPUNIT_ASSERT_THROW(dummyRecord.get(dummyPtr), ExceptionType);

  Dummy myDummy;
  WorkingDummyProxy workingProxy(&myDummy);
  ComponentDescription cd;
  cd.label_ = "";
  cd.type_ = "DummyProd";
  workingProxy.setProviderDescription(&cd);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working");

  dummyRecordImpl.add(workingDataKey, &workingProxy);

  dummyRecord.get("working", dummyPtr);
  CPPUNIT_ASSERT(!dummyPtr.failedToGet());
  CPPUNIT_ASSERT(dummyPtr.isValid());
  CPPUNIT_ASSERT(dummyPtr);
  (void)makeESValid(dummyPtr);

  CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

  const std::string workingString("working");
  dummyRecord.get(workingString, dummyPtr);
  CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

  edm::ESInputTag it_working("", "working");
  dummyRecord.get(it_working, dummyPtr);
  CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

  edm::ESInputTag it_prov("DummyProd", "working");
  dummyRecord.get(it_prov, dummyPtr);
  CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

  edm::ESInputTag it_bad("SmartProd", "working");
  CPPUNIT_ASSERT_THROW(dummyRecord.get(it_bad, dummyPtr), cms::Exception);

  //check if label is set
  cd.label_ = "foo";
  edm::ESInputTag it_label("foo", "working");
  dummyRecord.get(it_label, dummyPtr);
  CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

  edm::ESInputTag it_prov_bad("DummyProd", "working");
  CPPUNIT_ASSERT_THROW(dummyRecord.get(it_prov_bad, dummyPtr), cms::Exception);
}

namespace {
  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer(ESInputTag const& iTag) : m_token{esConsumes(iTag)} {}

    void prefetch(eventsetup::EventSetupRecordImpl const& iRec) const {
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        auto waitTask = edm::make_empty_waiting_task();
        waitTask->set_ref_count(2);
        iRec.prefetchAsync(waitTask.get(), proxies[i], nullptr, edm::ServiceToken{});
        waitTask->decrement_ref_count();
        waitTask->wait_for_all();
        if (waitTask->exceptionPtr()) {
          std::rethrow_exception(*waitTask->exceptionPtr());
        }
      }
    }

    ESGetToken<Dummy, DummyRecord> m_token;
  };
}  // namespace

namespace {
  struct SetupRecord {
    eventsetup::EventSetupRecordImpl dummyRecordImpl;
    DummyDataConsumer& consumer;
    //we need the DataKeys to stick around since references are being kept to them
    std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::DataProxy*>> proxies;

    SetupRecord(DummyDataConsumer& iConsumer,
                EventSetupRecordKey const& iKey,
                ActivityRegistry* iRegistry,
                std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::DataProxy*>> iProxies)
        : dummyRecordImpl(iKey, iRegistry), consumer(iConsumer), proxies(std::move(iProxies)) {
      for (auto const& d : proxies) {
        dummyRecordImpl.add(d.first, d.second);
      }

      ESRecordsToProxyIndices proxyIndices({iKey});
      std::vector<DataKey> dataKeys;
      dummyRecordImpl.fillRegisteredDataKeys(dataKeys);

      (void)proxyIndices.dataKeysInRecord(0, iKey, dataKeys, dummyRecordImpl.componentsForRegisteredDataKeys());

      iConsumer.updateLookup(proxyIndices);
      iConsumer.prefetch(dummyRecordImpl);
    }

    DummyRecord makeRecord() {
      DummyRecord ret;
      ret.setImpl(&dummyRecordImpl, 0, consumer.esGetTokenIndices(edm::Transition::Event), nullptr, false);
      return ret;
    }
  };
}  // namespace

void testEventsetupRecord::getHandleTest() {
  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  ESHandle<Dummy> dummyPtr;
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, &activityRegistry, {}};
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

    SetupRecord sr{consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.getHandle(consumer.m_token), ExceptionType);
  }
  Dummy myDummy;
  WorkingDummyProxy workingProxy(&myDummy);
  ComponentDescription cd;
  cd.label_ = "";
  cd.type_ = "DummyProd";
  workingProxy.setProviderDescription(&cd);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working");
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(!dummyPtr.failedToGet());
    CPPUNIT_ASSERT(dummyPtr.isValid());
    CPPUNIT_ASSERT(dummyPtr);

    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(not dummyRecord.getHandle(consumer.m_token));
    CPPUNIT_ASSERT_THROW(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
  }
  //check if label is set
  cd.label_ = "foo";
  {
    DummyDataConsumer consumer{edm::ESInputTag("foo", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();

    dummyPtr = dummyRecord.getHandle(consumer.m_token);
    CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT(not dummyRecord.getHandle(consumer.m_token));
    CPPUNIT_ASSERT_THROW(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
  }
}

void testEventsetupRecord::getWithTokenTest() {
  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, &activityRegistry, {}};

    DummyRecord dummyRecord = sr.makeRecord();

    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), NoDataExceptionType);
  }

  {
    DummyDataConsumer consumer{edm::ESInputTag("", "")};

    SetupRecord sr{consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), ExceptionType);
  }
  Dummy myDummy;
  WorkingDummyProxy workingProxy(&myDummy);
  ComponentDescription cd;
  cd.label_ = "";
  cd.type_ = "DummyProd";
  workingProxy.setProviderDescription(&cd);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working");
  {
    DummyDataConsumer consumer{edm::ESInputTag("", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);

    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);
    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), cms::Exception);
  }
  //check if label is set
  cd.label_ = "foo";
  {
    DummyDataConsumer consumer{edm::ESInputTag("foo", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    auto const& dummyData = dummyRecord.get(consumer.m_token);
    CPPUNIT_ASSERT(&dummyData == &myDummy);
  }
  {
    DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
    SetupRecord sr{
        consumer, dummyRecordKey_, &activityRegistry, {{dummyDataKey, &dummyProxy}, {workingDataKey, &workingProxy}}};

    DummyRecord dummyRecord = sr.makeRecord();
    CPPUNIT_ASSERT_THROW(dummyRecord.get(consumer.m_token), cms::Exception);
  }
}

void testEventsetupRecord::getNodataExpTest() {
  EventSetupRecordImpl recImpl(DummyRecord::keyForClass(), &activityRegistry);
  DummyRecord dummyRecord;
  dummyRecord.setImpl(&recImpl, 0, nullptr, nullptr, false);
  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");
  ESHandle<Dummy> dummyPtr;
  dummyRecord.get(dummyPtr);
  *dummyPtr;
}

void testEventsetupRecord::getExepTest() {
  eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};
  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  ESHandle<Dummy> dummyPtr;

  dummyRecordImpl.add(dummyDataKey, &dummyProxy);

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, nullptr, false);
  dummyRecord.get(dummyPtr);
  //CPPUNIT_ASSERT_THROW(dummyRecord.get(dummyPtr), ExceptionType);
}

void testEventsetupRecord::doGetTest() {
  eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};

  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, nullptr, false);
  CPPUNIT_ASSERT(!dummyRecord.doGet(dummyDataKey));

  dummyRecordImpl.add(dummyDataKey, &dummyProxy);

  //dummyRecord.doGet(dummyDataKey);
  CPPUNIT_ASSERT_THROW(dummyRecord.doGet(dummyDataKey), ExceptionType);

  Dummy myDummy;
  WorkingDummyProxy workingProxy(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working");

  dummyRecordImpl.add(workingDataKey, &workingProxy);

  CPPUNIT_ASSERT(dummyRecord.doGet(workingDataKey));
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
  FailingDummyProxy dummyProxy;

  ComponentDescription cd1;
  cd1.label_ = "foo1";
  cd1.type_ = "DummyProd1";
  cd1.isSource_ = false;
  cd1.isLooper_ = false;
  dummyProxy.setProviderDescription(&cd1);

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  std::vector<edm::eventsetup::DataKey> keys;
  dummyRecordImpl.fillRegisteredDataKeys(keys);
  CPPUNIT_ASSERT(keys.empty());

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, nullptr, false);

  std::vector<ComponentDescription const*> esproducers;
  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.empty());

  std::vector<DataKey> referencedDataKeys;
  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  auto referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.empty());
  CPPUNIT_ASSERT(referencedComponents.empty());

  dummyRecordImpl.add(dummyDataKey, &dummyProxy);

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
  WorkingDummyProxy workingProxy(&myDummy);

  ComponentDescription cd2;
  cd2.label_ = "foo2";
  cd2.type_ = "DummyProd2";
  cd2.isSource_ = true;
  cd2.isLooper_ = false;
  workingProxy.setProviderDescription(&cd2);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working");

  dummyRecordImpl.add(workingDataKey, &workingProxy);

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
  WorkingDummyProxy workingProxy3(&myDummy3);

  ComponentDescription cd3;
  cd3.label_ = "foo3";
  cd3.type_ = "DummyProd3";
  cd3.isSource_ = false;
  cd3.isLooper_ = true;
  workingProxy3.setProviderDescription(&cd3);

  const DataKey workingDataKey3(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working3");

  dummyRecordImpl.add(workingDataKey3, &workingProxy3);

  dummyRecordImpl.getESProducers(esproducers);
  CPPUNIT_ASSERT(esproducers.size() == 1);

  dummyRecordImpl.fillRegisteredDataKeys(referencedDataKeys);
  referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
  CPPUNIT_ASSERT(referencedDataKeys.size() == 3);
  CPPUNIT_ASSERT(referencedComponents.size() == 3);
  CPPUNIT_ASSERT(find(referencedDataKeys, referencedComponents, workingDataKey3) == &cd3);

  Dummy myDummy4;
  WorkingDummyProxy workingProxy4(&myDummy4);

  ComponentDescription cd4;
  cd4.label_ = "foo4";
  cd4.type_ = "DummyProd4";
  cd4.isSource_ = false;
  cd4.isLooper_ = false;
  workingProxy4.setProviderDescription(&cd4);

  const DataKey workingDataKey4(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "working4");

  dummyRecordImpl.add(workingDataKey4, &workingProxy4);

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
  eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};

  FailingDummyProxy dummyProxy;

  const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(), "");

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, nullptr, false);
  CPPUNIT_ASSERT(!dummyRecord.doGet(dummyDataKey));

  dummyRecordImpl.add(dummyDataKey, &dummyProxy);

  //typedef edm::eventsetup::MakeDataException<DummyRecord,Dummy> ExceptionType;
  dummyRecord.doGet(dummyDataKey);
  //CPPUNIT_ASSERT_THROW(dummyRecord.doGet(dummyDataKey), ExceptionType);
}

void testEventsetupRecord::proxyResetTest() {
  auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  EventSetupRecordProvider const* constRecordProvider = dummyProvider.get();

  DummyRecord dummyRecord;
  dummyRecord.setImpl(&constRecordProvider->firstRecordImpl(), 0, nullptr, nullptr, false);

  Dummy myDummy;
  std::shared_ptr<WorkingDummyProxy> workingProxy = std::make_shared<WorkingDummyProxy>(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "");

  std::shared_ptr<WorkingDummyProvider> wdProv = std::make_shared<WorkingDummyProvider>(workingDataKey, workingProxy);
  CPPUNIT_ASSERT(nullptr != wdProv.get());
  if (wdProv.get() == nullptr)
    return;  // To silence Coverity
  wdProv->createKeyedProxies(DummyRecord::keyForClass(), 1);
  dummyProvider->add(wdProv);

  //this causes the proxies to actually be placed in the Record
  edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
  dummyProvider->usePreferred(pref);

  CPPUNIT_ASSERT(dummyRecord.doGet(workingDataKey));

  edm::ESHandle<Dummy> hDummy;
  dummyRecord.get(hDummy);

  CPPUNIT_ASSERT(&myDummy == &(*hDummy));

  Dummy myDummy2;
  workingProxy->set(&myDummy2);

  //should not change
  dummyRecord.get(hDummy);
  CPPUNIT_ASSERT(&myDummy == &(*hDummy));
  CPPUNIT_ASSERT(!workingProxy->invalidateCalled());
  CPPUNIT_ASSERT(!workingProxy->invalidateTransientCalled());

  dummyProvider->resetProxies();
  CPPUNIT_ASSERT(workingProxy->invalidateCalled());
  CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled());
  dummyRecord.get(hDummy);
  CPPUNIT_ASSERT(&myDummy2 == &(*hDummy));
  CPPUNIT_ASSERT(!workingProxy->invalidateCalled());
  CPPUNIT_ASSERT(!workingProxy->invalidateTransientCalled());
}

void testEventsetupRecord::transientTest() {
  // NEEDS TO BE REWRITTEN WHEN WE FIX OR REMOVE THE TRANSIENT
  // PARTS OF THE EVENTSETUP CODE WHICH IS CURRENTLY DISABLED

  auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  DummyRecord dummyRecordNoConst;
  dummyRecordNoConst.setImpl(&dummyProvider->firstRecordImpl(), 0, nullptr, nullptr, false);
  EventSetupRecord const& dummyRecord = dummyRecordNoConst;

  eventsetup::EventSetupRecordImpl& nonConstDummyRecordImpl = *const_cast<EventSetupRecordImpl*>(dummyRecord.impl_);

  Dummy myDummy;
  std::shared_ptr<WorkingDummyProxy> workingProxy = std::make_shared<WorkingDummyProxy>(&myDummy);

  const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(), "");

  std::shared_ptr<WorkingDummyProvider> wdProv = std::make_shared<WorkingDummyProvider>(workingDataKey, workingProxy);
  wdProv->createKeyedProxies(DummyRecord::keyForClass(), 1);
  dummyProvider->add(wdProv);

  //this causes the proxies to actually be placed in the Record
  edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
  dummyProvider->usePreferred(pref);

  //do a transient access to see if it clears properly
  edm::ESTransientHandle<Dummy> hTDummy;
  CPPUNIT_ASSERT(hTDummy.transientAccessOnly);
  dummyRecord.get(hTDummy);

  CPPUNIT_ASSERT(&myDummy == &(*hTDummy));
  CPPUNIT_ASSERT(workingProxy->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled() == false);

  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingProxy->invalidateCalled());
  CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled());

  Dummy myDummy2;
  workingProxy->set(&myDummy2);

  //do non-transient access to make sure nothing resets now
  edm::ESHandle<Dummy> hDummy;
  dummyRecord.get(hDummy);

  dummyRecord.get(hDummy);
  CPPUNIT_ASSERT(&myDummy2 == &(*hDummy));
  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingProxy->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled() == false);

  //do another transient access which should not do a reset since we have a non-transient access outstanding
  dummyRecord.get(hDummy);
  dummyRecord.get(hTDummy);

  nonConstDummyRecordImpl.resetIfTransientInProxies();
  CPPUNIT_ASSERT(workingProxy->invalidateCalled() == false);
  CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled() == false);

  //Ask for a transient then a non transient to be sure we don't have an ordering problem
  {
    dummyProvider->resetProxies();
    Dummy myDummy3;
    workingProxy->set(&myDummy3);

    dummyRecord.get(hTDummy);
    dummyRecord.get(hDummy);

    CPPUNIT_ASSERT(&myDummy3 == &(*hDummy));
    CPPUNIT_ASSERT(&myDummy3 == &(*hTDummy));
    nonConstDummyRecordImpl.resetIfTransientInProxies();
    CPPUNIT_ASSERT(workingProxy->invalidateCalled() == false);
    CPPUNIT_ASSERT(workingProxy->invalidateTransientCalled() == false);
  }
}
