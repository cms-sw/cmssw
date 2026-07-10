/*
 *  eventsetuprecord_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/29/05.
 *  Changed by Viji on 06/07/2005
 */

#include "catch2/catch_all.hpp"

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/NoProductResolverException.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "FWCore/Framework/interface/HCTypeTag.h"

#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESModuleProducesInfo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESValidHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Utilities/interface/ESIndices.h"

#include "makeEmptyEventSetupImplForTest.h"

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

HCTYPETAG_HELPER_METHODS(Dummy)

class testEventsetupRecord {
public:
  template <typename R>
  static edm::ESConsumesCollectorT<R> makeCollector(ESConsumesInfo* const iConsumer, unsigned int iTransitionID) {
    return edm::ESConsumesCollectorT<R>(iConsumer, iTransitionID);
  }
  static const edm::eventsetup::EventSetupRecordImpl* impl(edm::eventsetup::EventSetupRecord const& iRecord) {
    return iRecord.impl_;
  }

  template <template <typename> typename H, typename T, typename R>
  static H<T> getHandleImpl(edm::eventsetup::EventSetupRecord const& iRecord, edm::ESGetToken<T, R> const& iToken) {
    return iRecord.getHandleImpl<H, T, R>(iToken);
  }
};
class FailingDummyResolver : public eventsetup::ESProductResolverTemplate<DummyRecord, Dummy> {
protected:
  const value_type* make(const record_type&, const DataKey&) final { return nullptr; }
  void const* getAfterPrefetchImpl() const final { return nullptr; }
};

class WorkingDummyResolver : public eventsetup::ESProductResolverTemplate<DummyRecord, Dummy> {
public:
  WorkingDummyResolver(const Dummy* iDummy)
      : data_(iDummy), invalidateCalled_(false), invalidateTransientCalled_(false) {}

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

class WorkingDummyProvider : public edm::eventsetup::ESProductResolverProvider {
public:
  WorkingDummyProvider(const edm::eventsetup::DataKey& iKey, std::shared_ptr<WorkingDummyResolver> iResolver)
      : m_key(iKey), m_resolver(iResolver) {
    usingRecord<DummyRecord>();
  }

  std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
    return std::vector<edm::eventsetup::ESModuleProducesInfo>(
        1,
        edm::eventsetup::ESModuleProducesInfo(edm::eventsetup::EventSetupRecordKey::makeKey<DummyRecord>(), m_key, 0));
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

namespace {
  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer(ESInputTag const& iTag) : m_token{esConsumes(iTag)} {}

    void prefetch(eventsetup::EventSetupRecordImpl const& iRec) const {
      auto const& resolvers = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != resolvers.size(); ++i) {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask waitTask{group};
        edm::ServiceToken token;
        iRec.prefetchAsync(WaitingTaskHolder(group, &waitTask), resolvers[i], nullptr, token, edm::ESParentContext{});
        waitTask.wait();
      }
    }

    ESGetToken<Dummy, DummyRecord> m_token;
  };

  struct DummyDataConsumerGeneric : public EDConsumerBase {
    explicit DummyDataConsumerGeneric(DataKey const& iKey)
        : m_token{esConsumes<>(eventsetup::EventSetupRecordKey::makeKey<DummyRecord>(), iKey)} {}

    void prefetch(eventsetup::EventSetupRecordImpl const& iRec) const {
      auto const& resolvers = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != resolvers.size(); ++i) {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask waitTask{group};
        edm::ServiceToken token;
        iRec.prefetchAsync(WaitingTaskHolder(group, &waitTask), resolvers[i], nullptr, token, edm::ESParentContext{});
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
    std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::ESProductResolver*>> resolvers;
    // same for ESParentContext
    ESParentContext pc_;

    SetupRecordT(CONSUMER& iConsumer,
                 EventSetupRecordKey const& iKey,
                 EventSetupImpl& iEventSetup,
                 ActivityRegistry* iRegistry,
                 std::vector<std::pair<edm::eventsetup::DataKey, edm::eventsetup::ESProductResolver*>> iResolvers)
        : dummyRecordImpl(iKey, iRegistry),
          eventSetupImpl_(iEventSetup),
          consumer(iConsumer),
          resolvers(std::move(iResolvers)) {
      for (auto const& d : resolvers) {
        dummyRecordImpl.add(d.first, d.second);
      }

      ESRecordsToProductResolverIndices resolverIndices({iKey});
      std::vector<DataKey> const& dataKeys = dummyRecordImpl.registeredDataKeys();

      (void)resolverIndices.dataKeysInRecord(0,
                                             iKey,
                                             dataKeys,
                                             dummyRecordImpl.componentsForRegisteredDataKeys(),
                                             dummyRecordImpl.produceMethodIDsForRegisteredDataKeys());

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
namespace {
  ComponentDescription const* find(std::vector<DataKey> const& iKeys,
                                   std::vector<ComponentDescription const*> const& iComp,
                                   DataKey const& iKey) {
    return iComp[std::lower_bound(iKeys.begin(), iKeys.end(), iKey) - iKeys.begin()];
  }
}  // namespace

TEST_CASE("EventSetupRecord", "[Framework][EventSetup]") {
  // Initialize member variables that were in the class
  oneapi::tbb::task_arena taskArena_(1);
  EventSetupImpl eventSetupImpl_ = makeEmptyEventSetupImplForTest();
  EventSetupRecordKey dummyRecordKey_ = EventSetupRecordKey::makeKey<DummyRecord>();

  SECTION("resolverTest") {
    eventsetup::EventSetupRecordImpl dummyRecord{dummyRecordKey_, &activityRegistry};

    FailingDummyResolver dummyResolver;

    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    REQUIRE(nullptr == dummyRecord.find(dummyDataKey));

    dummyRecord.add(dummyDataKey, &dummyResolver);
    REQUIRE(&dummyResolver == dummyRecord.find(dummyDataKey));

    const DataKey dummyFredDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "fred");
    REQUIRE(nullptr == dummyRecord.find(dummyFredDataKey));
  }

  SECTION("getHandleTest") {
    FailingDummyResolver dummyResolver;

    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    ESHandle<Dummy> dummyPtr;
    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};

      SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};
      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE(not dummyRecord.getHandle(consumer.m_token));
      dummyPtr = dummyRecord.getHandle(consumer.m_token);
      REQUIRE(not dummyPtr.isValid());
      REQUIRE(not dummyPtr);
      REQUIRE(dummyPtr.failedToGet());
      REQUIRE_THROWS_AS(*dummyPtr, NoProductResolverException);
      REQUIRE_THROWS_AS(makeESValid(dummyPtr), cms::Exception);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};

      SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();
      REQUIRE_THROWS_AS(dummyRecord.getHandle(consumer.m_token), ExceptionType);
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
      REQUIRE(!dummyPtr.failedToGet());
      REQUIRE(dummyPtr.isValid());
      REQUIRE(dummyPtr);

      REQUIRE(&(*dummyPtr) == &myDummy);
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
      REQUIRE(&(*dummyPtr) == &myDummy);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
      SetupRecord sr{consumer,
                     dummyRecordKey_,
                     eventSetupImpl_,
                     &activityRegistry,
                     {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE(not dummyRecord.getHandle(consumer.m_token));
      REQUIRE_THROWS_AS(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
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
      REQUIRE(&(*dummyPtr) == &myDummy);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
      SetupRecord sr{consumer,
                     dummyRecordKey_,
                     eventSetupImpl_,
                     &activityRegistry,
                     {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE(not dummyRecord.getHandle(consumer.m_token));
      REQUIRE_THROWS_AS(*dummyRecord.getHandle(consumer.m_token), cms::Exception);
    }
  }

  SECTION("getWithTokenTest") {
    FailingDummyResolver dummyResolver;

    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};

      SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE_THROWS_AS(dummyRecord.get(consumer.m_token), NoProductResolverException);
    }

    {
      DummyDataConsumer consumer{edm::ESInputTag("", "")};

      SetupRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();
      REQUIRE_THROWS_AS(dummyRecord.get(consumer.m_token), ExceptionType);
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

      REQUIRE(&dummyData == &myDummy);
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
      REQUIRE(&dummyData == &myDummy);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("SmartProd", "working")};
      SetupRecord sr{consumer,
                     dummyRecordKey_,
                     eventSetupImpl_,
                     &activityRegistry,
                     {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();
      REQUIRE_THROWS_AS(dummyRecord.get(consumer.m_token), cms::Exception);
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
      REQUIRE(&dummyData == &myDummy);
    }
    {
      DummyDataConsumer consumer{edm::ESInputTag("DummyProd", "working")};
      SetupRecord sr{consumer,
                     dummyRecordKey_,
                     eventSetupImpl_,
                     &activityRegistry,
                     {{dummyDataKey, &dummyResolver}, {workingDataKey, &workingResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();
      REQUIRE_THROWS_AS(dummyRecord.get(consumer.m_token), cms::Exception);
    }
  }

  SECTION("getNodataExpTest") {
    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    edm::ESConsumesInfo consumesInfo;
    edm::ESConsumesCollectorT<DummyRecord> cc = testEventsetupRecord::makeCollector<DummyRecord>(
        &consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
    auto token = cc.consumes<Dummy>();
    std::vector<edm::ESResolverIndex> getTokenIndices{ESResolverIndex::noResolverConfigured()};

    EventSetupRecordImpl recImpl(DummyRecord::keyForClass(), &activityRegistry);
    DummyRecord dummyRecord;
    ESParentContext pc;
    dummyRecord.setImpl(&recImpl, 0, getTokenIndices.data(), &eventSetupImpl_, &pc);

    REQUIRE_THROWS_AS(*dummyRecord.getHandle(token), NoProductResolverException);
  }

  SECTION("doGetTest") {
    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    {
      DummyDataConsumerGeneric consumer{dummyDataKey};

      SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE(!dummyRecord.doGet(consumer.m_token));
    }

    FailingDummyResolver dummyResolver;

    {
      DummyDataConsumerGeneric consumer{dummyDataKey};

      SetupGenericRecord sr{
          consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();
      REQUIRE_THROWS_AS(dummyRecord.doGet(consumer.m_token), ExceptionType);
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
      REQUIRE(dummyRecord.doGet(consumer.m_token));
    }
  }

  SECTION("introspectionTest") {
    eventsetup::EventSetupRecordImpl dummyRecordImpl{dummyRecordKey_, &activityRegistry};
    FailingDummyResolver dummyResolver;

    ComponentDescription cd1;
    cd1.label_ = "foo1";
    cd1.type_ = "DummyProd1";
    cd1.isSource_ = false;
    cd1.isLooper_ = false;
    dummyResolver.setProviderDescription(&cd1);

    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

    std::vector<edm::eventsetup::DataKey> keys = dummyRecordImpl.registeredDataKeys();
    REQUIRE(keys.empty());

    DummyRecord dummyRecord;
    ESParentContext pc;
    dummyRecord.setImpl(&dummyRecordImpl, 0, nullptr, &eventSetupImpl_, &pc);

    std::vector<ComponentDescription const*> esproducers;
    dummyRecordImpl.getESProducers(esproducers);
    REQUIRE(esproducers.empty());

    std::vector<DataKey> referencedDataKeys = dummyRecordImpl.registeredDataKeys();
    auto referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
    REQUIRE(referencedDataKeys.empty());
    REQUIRE(referencedComponents.empty());

    dummyRecordImpl.add(dummyDataKey, &dummyResolver);

    dummyRecord.fillRegisteredDataKeys(keys);
    REQUIRE(1 == keys.size());

    dummyRecordImpl.getESProducers(esproducers);
    REQUIRE(esproducers.size() == 1);
    REQUIRE(esproducers[0] == &cd1);

    referencedDataKeys = dummyRecordImpl.registeredDataKeys();
    referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
    REQUIRE(referencedDataKeys.size() == 1);
    REQUIRE(referencedComponents.size() == 1);
    REQUIRE(referencedComponents[0] == &cd1);
    REQUIRE(find(referencedDataKeys, referencedComponents, dummyDataKey) == &cd1);

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
    REQUIRE(2 == keys.size());

    dummyRecordImpl.getESProducers(esproducers);
    REQUIRE(esproducers.size() == 1);

    referencedDataKeys = dummyRecordImpl.registeredDataKeys();
    referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
    REQUIRE(referencedDataKeys.size() == 2);
    REQUIRE(referencedComponents.size() == 2);
    REQUIRE(find(referencedDataKeys, referencedComponents, workingDataKey) == &cd2);

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
    REQUIRE(esproducers.size() == 1);

    referencedDataKeys = dummyRecordImpl.registeredDataKeys();
    referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
    REQUIRE(referencedDataKeys.size() == 3);
    REQUIRE(referencedComponents.size() == 3);
    REQUIRE(find(referencedDataKeys, referencedComponents, workingDataKey3) == &cd3);

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
    REQUIRE(esproducers.size() == 2);
    REQUIRE(esproducers[1] == &cd4);

    referencedDataKeys = dummyRecordImpl.registeredDataKeys();
    referencedComponents = dummyRecordImpl.componentsForRegisteredDataKeys();
    REQUIRE(referencedDataKeys.size() == 4);
    REQUIRE(referencedComponents.size() == 4);
    REQUIRE(find(referencedDataKeys, referencedComponents, workingDataKey4) == &cd4);

    dummyRecordImpl.clearResolvers();
    dummyRecord.fillRegisteredDataKeys(keys);
    REQUIRE(0 == keys.size());
  }

  SECTION("doGetExepTest") {
    const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");
    {
      DummyDataConsumerGeneric consumer{dummyDataKey};

      SetupGenericRecord sr{consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE(!dummyRecord.doGet(consumer.m_token));
    }

    {
      FailingDummyResolver dummyResolver;

      const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyResolver::value_type>(), "");

      DummyDataConsumerGeneric consumer{dummyDataKey};

      SetupGenericRecord sr{
          consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{dummyDataKey, &dummyResolver}}};

      DummyRecord dummyRecord = sr.makeRecord();

      REQUIRE_THROWS_AS(dummyRecord.doGet(consumer.m_token), ExceptionType);
    }
  }

  SECTION("resolverResetTest") {
    auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

    Dummy myDummy;
    std::shared_ptr<WorkingDummyResolver> workingResolver = std::make_shared<WorkingDummyResolver>(&myDummy);

    const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "");
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    SetupRecord sr{
        consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{workingDataKey, workingResolver.get()}}};
    DummyRecord dummyRecord = sr.makeRecord();

    edm::ESConsumesInfo consumesInfo;
    edm::ESConsumesCollectorT<DummyRecord> cc = testEventsetupRecord::makeCollector<DummyRecord>(
        &consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
    auto token = cc.consumes<Dummy>();
    std::vector<edm::ESResolverIndex> getTokenIndices{edm::ESResolverIndex(0)};

    std::shared_ptr<WorkingDummyProvider> wdProv =
        std::make_shared<WorkingDummyProvider>(workingDataKey, workingResolver);
    REQUIRE(nullptr != wdProv.get());
    if (wdProv.get() == nullptr)
      return;  // To silence Coverity
    wdProv->createKeyedResolvers(DummyRecord::keyForClass(), 1);
    dummyProvider->add(wdProv);

    //this causes the resolvers to actually be placed in the Record
    edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
    dummyProvider->usePreferred(pref);

    edm::ESHandle<Dummy> hDummy = dummyRecord.getHandle(token);

    REQUIRE(&myDummy == &(*hDummy));

    Dummy myDummy2;
    workingResolver->set(&myDummy2);

    //should not change
    hDummy = dummyRecord.getHandle(token);
    REQUIRE(&myDummy == &(*hDummy));
    REQUIRE(!workingResolver->invalidateCalled());
    REQUIRE(!workingResolver->invalidateTransientCalled());

    dummyProvider->resetResolvers();
    REQUIRE(workingResolver->invalidateCalled());
    REQUIRE(workingResolver->invalidateTransientCalled());
    consumer.prefetch(sr.dummyRecordImpl);
    hDummy = dummyRecord.getHandle(token);
    REQUIRE(&myDummy2 == &(*hDummy));
    REQUIRE(!workingResolver->invalidateCalled());
    REQUIRE(!workingResolver->invalidateTransientCalled());
  }

  SECTION("transientTest") {
    // NEEDS TO BE REWRITTEN WHEN WE FIX OR REMOVE THE TRANSIENT
    // PARTS OF THE EVENTSETUP CODE WHICH IS CURRENTLY DISABLED

    auto dummyProvider = std::make_unique<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

    Dummy myDummy;
    std::shared_ptr<WorkingDummyResolver> workingResolver = std::make_shared<WorkingDummyResolver>(&myDummy);

    const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyResolver::value_type>(), "");
    DummyDataConsumer consumer{edm::ESInputTag("", "")};
    SetupRecord sr{
        consumer, dummyRecordKey_, eventSetupImpl_, &activityRegistry, {{workingDataKey, workingResolver.get()}}};
    DummyRecord dummyRecordNoConst = sr.makeRecord();
    EventSetupRecord const& dummyRecord = dummyRecordNoConst;

    edm::ESConsumesInfo consumesInfo;
    edm::ESConsumesCollectorT<DummyRecord> cc = testEventsetupRecord::makeCollector<DummyRecord>(
        &consumesInfo, static_cast<unsigned int>(edm::Transition::Event));
    auto token = cc.consumes<Dummy>();
    std::vector<edm::ESResolverIndex> getTokenIndices{edm::ESResolverIndex(0)};

    eventsetup::EventSetupRecordImpl& nonConstDummyRecordImpl =
        *const_cast<EventSetupRecordImpl*>(testEventsetupRecord::impl(dummyRecord));

    std::shared_ptr<WorkingDummyProvider> wdProv =
        std::make_shared<WorkingDummyProvider>(workingDataKey, workingResolver);
    wdProv->createKeyedResolvers(DummyRecord::keyForClass(), 1);
    dummyProvider->add(wdProv);

    //this causes the resolvers to actually be placed in the Record
    edm::eventsetup::EventSetupRecordProvider::DataToPreferredProviderMap pref;
    dummyProvider->usePreferred(pref);

    //do a transient access to see if it clears properly
    edm::ESTransientHandle<Dummy> hTDummy;
    REQUIRE(hTDummy.transientAccessOnly);
    hTDummy = testEventsetupRecord::getHandleImpl<edm::ESTransientHandle>(dummyRecord, token);

    REQUIRE(&myDummy == &(*hTDummy));
    REQUIRE(workingResolver->invalidateCalled() == false);
    REQUIRE(workingResolver->invalidateTransientCalled() == false);

    nonConstDummyRecordImpl.resetIfTransientInResolvers();
    REQUIRE(workingResolver->invalidateCalled());
    REQUIRE(workingResolver->invalidateTransientCalled());

    Dummy myDummy2;
    workingResolver->set(&myDummy2);

    //do non-transient access to make sure nothing resets now
    consumer.prefetch(sr.dummyRecordImpl);
    edm::ESHandle<Dummy> hDummy = testEventsetupRecord::getHandleImpl<edm::ESHandle>(dummyRecord, token);

    hDummy = testEventsetupRecord::getHandleImpl<edm::ESHandle>(dummyRecord, token);
    REQUIRE(&myDummy2 == &(*hDummy));
    nonConstDummyRecordImpl.resetIfTransientInResolvers();
    REQUIRE(workingResolver->invalidateCalled() == false);
    REQUIRE(workingResolver->invalidateTransientCalled() == false);

    //do another transient access which should not do a reset since we have a non-transient access outstanding
    consumer.prefetch(sr.dummyRecordImpl);
    hDummy = testEventsetupRecord::getHandleImpl<edm::ESHandle>(dummyRecord, token);
    hTDummy = testEventsetupRecord::getHandleImpl<edm::ESTransientHandle>(dummyRecord, token);

    nonConstDummyRecordImpl.resetIfTransientInResolvers();
    REQUIRE(workingResolver->invalidateCalled() == false);
    REQUIRE(workingResolver->invalidateTransientCalled() == false);

    //Ask for a transient then a non transient to be sure we don't have an ordering problem
    {
      dummyProvider->resetResolvers();
      Dummy myDummy3;
      workingResolver->set(&myDummy3);

      consumer.prefetch(sr.dummyRecordImpl);
      hDummy = testEventsetupRecord::getHandleImpl<edm::ESHandle>(dummyRecord, token);
      hTDummy = testEventsetupRecord::getHandleImpl<edm::ESTransientHandle>(dummyRecord, token);

      REQUIRE(&myDummy3 == &(*hDummy));
      REQUIRE(&myDummy3 == &(*hTDummy));
      nonConstDummyRecordImpl.resetIfTransientInResolvers();
      REQUIRE(workingResolver->invalidateCalled() == false);
      REQUIRE(workingResolver->invalidateTransientCalled() == false);
    }
  }
}
