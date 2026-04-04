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
// in tests but in a normal cmsRun job this is done through the controller.

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/NoProductResolverException.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

#include "FWCore/Framework/src/SynchronousEventSetupsController.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyEventSetupRecordRetriever.h"
#include "FWCore/Framework/test/DummyESProductResolverProvider.h"
#include "FWCore/Framework/test/DummyRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESProductTag.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "makeEmptyEventSetupImplForTest.h"

#include "catch2/catch_all.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>

using namespace edm;
using namespace edm::eventsetup;
using edm::eventsetup::test::DummyData;
using edm::eventsetup::test::DummyESProductResolverProvider;

class testEventsetup {
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
    return iRecord.getHandleImpl(iToken);
  }

  static void setKeyIters(edm::EventSetupImpl& iImpl,
                          std::vector<edm::eventsetup::EventSetupRecordKey>::const_iterator iBegin,
                          std::vector<edm::eventsetup::EventSetupRecordKey>::const_iterator iEnd) {
    iImpl.setKeyIters(iBegin, iEnd);
  }

  static void addRecordImpl(edm::EventSetupImpl& iEventSetupImpl, edm::eventsetup::EventSetupRecordImpl& iRecordImpl) {
    iEventSetupImpl.addRecordImpl(iRecordImpl);
  }
};

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
    }
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
    SetMayConsumeProducer(bool iSucceed,
                          char const* conditionalModuleLabel,
                          char const* conditionalProductLabel,
                          char const* producedProductLabel)
        : succeed_(iSucceed),
          conditionalModuleLabel_(conditionalModuleLabel),
          conditionalProductLabel_(conditionalProductLabel),
          producedProductLabel_(producedProductLabel) {
      setWhatProduced(this, producedProductLabel)
          .setMayConsume(
              token_,
              [this](auto& get, edm::ESTransientHandle<edm::eventsetup::test::DummyData> const& handle) {
                if (succeed_) {
                  return get(conditionalModuleLabel_, conditionalProductLabel_);
                }
                return get.nothing();
              },
              edm::ESProductTag<edm::eventsetup::test::DummyData, DummyRecord>("", ""));
    }
    std::unique_ptr<edm::eventsetup::test::DummyData> produce(const DummyRecord& iRecord) {
      auto const& data = iRecord.getHandle(token_);
      if (succeed_) {
        return std::make_unique<edm::eventsetup::test::DummyData>(*data);
      }
      REQUIRE(!data.isValid());
      return std::unique_ptr<edm::eventsetup::test::DummyData>();
    }

  private:
    edm::ESGetToken<edm::eventsetup::test::DummyData, DummyRecord> token_;
    bool succeed_;
    char const* conditionalModuleLabel_;
    char const* conditionalProductLabel_;
    char const* producedProductLabel_;
  };

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

  struct DummyDataConsumer : public EDConsumerBase {
    explicit DummyDataConsumer(ESInputTag const& iTag) : m_token{esConsumes(iTag)} {}

    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& resolvers = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != resolvers.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          rec->prefetchAsync(
              WaitingTaskHolder(group, &waitTask), resolvers[i], &iImpl, edm::ServiceToken{}, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }

    ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> m_token;
    ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> m_tokenUninitialized;
  };
}  // namespace

TEST_CASE("EventSetup", "[Framework][EventSetup]") {
  auto m_scheduler = std::make_unique<edm::ThreadsController>(1);
  DummyData kGood{1};
  DummyData kBad{0};

  edm::ActivityRegistry activityRegistry;

  SECTION("constructTest") { edm::ActivityRegistry activityRegistry; }

  SECTION("constructTest") {
    eventsetup::EventSetupProvider provider(&activityRegistry);
    const Timestamp time(1);
    const IOVSyncValue timestamp(time);
    bool newEventSetupImpl = false;
    auto eventSetupImpl = provider.eventSetupForInstance(timestamp, newEventSetupImpl);
    REQUIRE(non_null(eventSetupImpl.get()));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
  }

  // Note there is a similar test in dependentrecord_t.catch2.cc
  // named getTest() that tests get and tryToGet using an EventSetupProvider.
  // No need to repeat that test here. The next two just test EventSetupImpl
  // at the lowest level.

  SECTION("getTest") {
    EventSetupImpl eventSetupImpl = makeEmptyEventSetupImplForTest();
    std::vector<eventsetup::EventSetupRecordKey> keys;
    EventSetupRecordKey key = EventSetupRecordKey::makeKey<DummyRecord>();
    keys.push_back(key);
    testEventsetup::setKeyIters(eventSetupImpl, keys.begin(), keys.end());
    EventSetupRecordImpl dummyRecordImpl{key, &activityRegistry};
    testEventsetup::addRecordImpl(eventSetupImpl, dummyRecordImpl);
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(eventSetupImpl, 0, nullptr, pc);
    const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
    REQUIRE(&dummyRecordImpl == testEventsetup::impl(gottenRecord));
  }

  SECTION("tryToGetTest") {
    EventSetupImpl eventSetupImpl = makeEmptyEventSetupImplForTest();
    std::vector<eventsetup::EventSetupRecordKey> keys;
    EventSetupRecordKey key = EventSetupRecordKey::makeKey<DummyRecord>();
    keys.push_back(key);
    testEventsetup::setKeyIters(eventSetupImpl, keys.begin(), keys.end());
    EventSetupRecordImpl dummyRecordImpl{key, &activityRegistry};
    testEventsetup::addRecordImpl(eventSetupImpl, dummyRecordImpl);
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(eventSetupImpl, 0, nullptr, pc);
    std::optional<DummyRecord> gottenRecord = eventSetup.tryToGet<DummyRecord>();
    REQUIRE(gottenRecord);
    REQUIRE(&dummyRecordImpl == testEventsetup::impl(gottenRecord.value()));
  }

  SECTION("getExcTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE_THROWS_AS(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
  }

  SECTION("recordValidityTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE(!eventSetup1.tryToGet<DummyRecord>().has_value());

    const Timestamp time_2(2);
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(time_2), IOVSyncValue(Timestamp(3))));
    {
      controller.eventSetupForInstance(IOVSyncValue(time_2));
      edm::ESParentContext pc;
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
      eventSetup.get<DummyRecord>();
      REQUIRE(eventSetup.tryToGet<DummyRecord>().has_value());
    }

    {
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(3)));
      edm::ESParentContext pc;
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
      eventSetup.get<DummyRecord>();
      REQUIRE(eventSetup.tryToGet<DummyRecord>().has_value());
    }
    {
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(4)));
      edm::ESParentContext pc;
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
      REQUIRE(!eventSetup.tryToGet<DummyRecord>().has_value());
    }
  }

  SECTION("recordValidityExcTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE_THROWS_AS(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
  }

  SECTION("recordValidityNoFinderTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE(!eventSetup.tryToGet<DummyRecord>().has_value());
  }

  SECTION("recordValidityNoFinderExcTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE_THROWS_AS(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
  }

  //create an instance of the register
  static eventsetup::RecordDependencyRegister<DummyRecord> s_factory;

  SECTION("recordValidityResolverNoFinderTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    provider.add(std::make_shared<DummyESProductResolverProvider>());

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE(!eventSetup.tryToGet<DummyRecord>().has_value());
  }

  SECTION("recordValidityResolverNoFinderExcTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    provider.add(std::make_shared<DummyESProductResolverProvider>());

    Timestamp time_1(1);
    controller.eventSetupForInstance(IOVSyncValue(time_1));
    edm::ESParentContext pc;
    const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE_THROWS_AS(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
  }

  SECTION("producerConflictTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, false);

    {
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>();
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>();
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    //checking for conflicts is delayed until first eventSetupForInstance
    REQUIRE_THROWS_AS(controller.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue()), cms::Exception);
  }

  SECTION("sourceConflictTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);

    {
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>();
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>();
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    //checking for conflicts is delayed until first eventSetupForInstance
    REQUIRE_THROWS_AS(controller.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue()), cms::Exception);
  }

  SECTION("twoSourceTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
    {
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>();
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      auto dummyProv = std::make_shared<edm::DummyEventSetupRecordRetriever>();
      std::shared_ptr<eventsetup::ESProductResolverProvider> providerPtr(dummyProv);
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
    REQUIRE(!eventSetup3.tryToGet<DummyRecord>().has_value());
    REQUIRE(!eventSetup3.tryToGet<DummyEventSetupRecord>().has_value());
    controller.eventSetupForInstance(IOVSyncValue(Timestamp(3)));
    const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, pc);
    REQUIRE(!eventSetup4.tryToGet<DummyRecord>().has_value());
    REQUIRE(eventSetup4.tryToGet<DummyEventSetupRecord>().has_value());
    eventSetup4.get<DummyEventSetupRecord>();
  }
  SECTION("provenanceTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    try {
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test11");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testLabel", 1, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test22");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      edm::ESParentContext pc;
      const edm::EventSetup eventSetup(provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       pc);
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);

      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testLabel");
    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("getDataWithESGetTokenTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    try {
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test11");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testTwo", 1, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test22");
        ps.addParameter<std::string>("appendToDataLabel", "blah");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 100, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test22");
        ps.addParameter<std::string>("appendToDataLabel", "blahblah");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
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
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(true, "", "", "setMayConsumeSucceed");
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
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(false, "", "", "setMayConsumeFail");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description(
            "SetMayConsumeProducer", "setMayConsumeWithModuleLabel", 5, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "thisIsNotUsed");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv =
            std::make_shared<SetMayConsumeProducer>(true, "testTwo", "blah", "productLabelForProducerWithModuleLabel");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description(
            "SetMayConsumeProducer", "setMayConsumeWithModuleLabelThatDoesntExist", 5, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "thisIsNotUsed");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(
            true, "doesNotExist", "blah", "productLabelForProducerWithModuleLabelThatDoesntExist");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description(
            "SetMayConsumeProducer", "setMayConsumeWithProductLabelThatDoesntExist", 5, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "thisIsNotUsed");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(
            true, "testTwo", "doesNotExist", "productLabelForProducerWithProductLabelThatDoesntExist");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description(
            "SetMayConsumeProducer", "setMayConsumeWithUnlabeledModuleLabel", 101, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "thisIsNotUsed");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(
            true, "DummyESProductResolverProvider", "blahblah", "productLabelForProducerWithMayConsumesUnlabeledCase");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description(
            "SetMayConsumeProducer", "setMayConsumeWithUnlabeledModuleLabel", 102, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "thisIsNotUsed");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<SetMayConsumeProducer>(
            true, "doesNotExist", "blahblah", "productLabelForProducerWithMayConsumesUnlabeledCaseNonexistent");
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }

      edm::ESParentContext pc;
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      {
        DummyDataConsumer consumer{edm::ESInputTag("", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        auto const& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kGood.value_ == data.value_);
        bool uninitializedTokenThrewException = false;
        try {
          (void)eventSetup.getData(consumer.m_tokenUninitialized);
        } catch (cms::Exception& ex) {
          uninitializedTokenThrewException = true;
          REQUIRE(ex.category() == "InvalidESGetToken");
        }
        REQUIRE(uninitializedTokenThrewException);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kBad.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        auto const& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kGood.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("testTwo", "DoesNotExist")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        REQUIRE_THROWS_AS(eventSetup.getData(consumer.m_token), edm::eventsetup::NoProductResolverException);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        REQUIRE_THROWS_AS(eventSetup.getData(consumer.m_token), cms::Exception);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "consumes")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kBad.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "consumesFrom")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kBad.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "setMayConsumeFail")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        REQUIRE_THROWS_AS(eventSetup.getData(consumer.m_token), edm::eventsetup::MakeDataException);
      }
      {
        DummyDataConsumer consumer{edm::ESInputTag("", "setMayConsumeSucceed")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kBad.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "productLabelForProducerWithModuleLabel")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kGood.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "productLabelForProducerWithModuleLabelThatDoesntExist")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        REQUIRE_THROWS_AS(consumer.prefetch(provider.eventSetupImpl()), edm::eventsetup::NoProductResolverException);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "productLabelForProducerWithProductLabelThatDoesntExist")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        REQUIRE_THROWS_AS(consumer.prefetch(provider.eventSetupImpl()), edm::eventsetup::NoProductResolverException);
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "productLabelForProducerWithMayConsumesUnlabeledCase")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        const DummyData& data = eventSetup.getData(consumer.m_token);
        REQUIRE(kGood.value_ == data.value_);
      }

      {
        DummyDataConsumer consumer{
            edm::ESInputTag("", "productLabelForProducerWithMayConsumesUnlabeledCaseNonexistent")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        REQUIRE_THROWS_AS(consumer.prefetch(provider.eventSetupImpl()), edm::eventsetup::NoProductResolverException);
      }

    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("getHandleWithESGetTokenTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    edm::ESParentContext pc;
    try {
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test11");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testTwo", 1, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test22");
        ps.addParameter<std::string>("appendToDataLabel", "blah");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
        dummyProv->setDescription(description);
        dummyProv->setAppendToDataLabel(ps);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      {
        DummyDataConsumer consumer{edm::ESInputTag("", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());

        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testTwo");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kBad.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testOne");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testTwo");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        REQUIRE(not eventSetup.getHandle(consumer.m_token));
        REQUIRE_THROWS_AS(*eventSetup.getHandle(consumer.m_token), cms::Exception);
      }

    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("getTransientHandleWithESGetTokenTest") {
    using edm::eventsetup::test::DummyData;
    using edm::eventsetup::test::DummyESProductResolverProvider;
    DummyData kGood{1};
    DummyData kBad{0};

    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    edm::ESParentContext pc;
    try {
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test11");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testTwo", 1, false);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test22");
        ps.addParameter<std::string>("appendToDataLabel", "blah");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
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
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                         static_cast<unsigned int>(edm::Transition::Event),
                                         consumer.esGetTokenIndices(edm::Transition::Event),
                                         pc};
        edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testTwo");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                         static_cast<unsigned int>(edm::Transition::Event),
                                         consumer.esGetTokenIndices(edm::Transition::Event),
                                         pc};
        edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
        REQUIRE(kBad.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testOne");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("testTwo", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                         static_cast<unsigned int>(edm::Transition::Event),
                                         consumer.esGetTokenIndices(edm::Transition::Event),
                                         pc};
        edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
        const edm::eventsetup::ComponentDescription* desc = data.description();
        REQUIRE(desc->label_ == "testTwo");
      }

      {
        DummyDataConsumer consumer{edm::ESInputTag("DoesNotExist", "blah")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                         static_cast<unsigned int>(edm::Transition::Event),
                                         consumer.esGetTokenIndices(edm::Transition::Event),
                                         pc};
        edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
        REQUIRE(not data);
        REQUIRE_THROWS_AS(*data, cms::Exception);
      }

    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("sourceProducerResolutionTest") {
    {
      SynchronousEventSetupsController controller;
      edm::ParameterSet pset = createDummyPset();
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

      std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
      dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
      provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

      edm::ESParentContext pc;
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 1, false);
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const EventSetup eventSetup{provider.eventSetupImpl(),
                                  static_cast<unsigned int>(edm::Transition::Event),
                                  consumer.esGetTokenIndices(edm::Transition::Event),
                                  pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
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
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, false);
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 1, true);
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      ESParentContext pc;
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
    }
  }

  SECTION("preferTest") {
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
        //default means use all resolvers
        preferInfo[ComponentDescription("DummyESProductResolverProvider", "", ComponentDescription::unknownID(), false)] =
            recordToData;
        provider.setPreferredProviderInfo(preferInfo);
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "bad", 0, false);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 1, false);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
      }

      //sources
      {
        SynchronousEventSetupsController controller;
        EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);
        provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

        EventSetupProvider::PreferredProviderInfo preferInfo;
        EventSetupProvider::RecordToDataMap recordToData;
        //default means use all resolvers
        preferInfo[ComponentDescription("DummyESProductResolverProvider", "", ComponentDescription::unknownID(), false)] =
            recordToData;
        provider.setPreferredProviderInfo(preferInfo);
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "bad", 1, true);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
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
        preferInfo[ComponentDescription("DummyESProductResolverProvider", "", ComponentDescription::unknownID(), false)] =
            recordToData;
        provider.setPreferredProviderInfo(preferInfo);
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kGood);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        {
          edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "bad", 1, true);
          auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
          dummyProv->setDescription(description);
          provider.add(dummyProv);
        }
        controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
        DummyDataConsumer consumer{edm::ESInputTag("", "")};
        consumer.updateLookup(provider.recordsToResolverIndices());
        consumer.prefetch(provider.eventSetupImpl());
        EventSetup eventSetup{provider.eventSetupImpl(),
                              static_cast<unsigned int>(edm::Transition::Event),
                              consumer.esGetTokenIndices(edm::Transition::Event),
                              pc};
        edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
        REQUIRE(kGood.value_ == data->value_);
      }

    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("introspectionTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    edm::ESParentContext pc;
    try {
      {
        edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
        edm::ParameterSet ps;
        ps.addParameter<std::string>("name", "test11");
        ps.registerIt();
        description.pid_ = ps.id();
        auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(2)));
      {
        EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};

        REQUIRE(eventSetup.recordIsProvidedByAModule(dummyRecordKey));
        std::vector<edm::eventsetup::EventSetupRecordKey> recordKeys;
        eventSetup.fillAvailableRecordKeys(recordKeys);
        REQUIRE(1 == recordKeys.size());
        REQUIRE(dummyRecordKey == recordKeys[0]);
        auto record = eventSetup.find(recordKeys[0]);
        REQUIRE(record.has_value());
      }
      // Intentionally an out of range sync value so the IOV is invalid
      // to test the find function with a record that exists in the
      // EventSetupImpl but has a null pointer.
      controller.eventSetupForInstance(IOVSyncValue(Timestamp(4)));
      {
        EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};

        REQUIRE(eventSetup.recordIsProvidedByAModule(dummyRecordKey));
        std::vector<edm::eventsetup::EventSetupRecordKey> recordKeys;
        eventSetup.fillAvailableRecordKeys(recordKeys);
        REQUIRE(0 == recordKeys.size());
        auto record = eventSetup.find(dummyRecordKey);
        REQUIRE(!record.has_value());

        // Just to try all cases test find with a record type not in the EventSetupImpl
        // at all.
        EventSetupRecordKey dummyRecordKey1 = EventSetupRecordKey::makeKey<DummyEventSetupRecord>();
        auto record1 = eventSetup.find(dummyRecordKey1);
        REQUIRE(!record1.has_value());
      }
    } catch (const cms::Exception& iException) {
      std::cout << "caught " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("iovExtensionTest") {
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
      REQUIRE(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
    }
    {
      controller.eventSetupForInstance(IOVSyncValue{Timestamp(3)});
      EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
      eventSetup.get<DummyRecord>();
      REQUIRE(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
    }
    //extending the IOV should not cause the cache to be reset
    finder->setInterval(ValidityInterval(IOVSyncValue{Timestamp{2}}, IOVSyncValue{Timestamp{4}}));
    {
      controller.eventSetupForInstance(IOVSyncValue{Timestamp(4)});
      EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
      REQUIRE(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
    }

    //this is a new IOV so should get cache reset
    finder->setInterval(ValidityInterval(IOVSyncValue{Timestamp{5}}, IOVSyncValue{Timestamp{6}}));
    {
      controller.eventSetupForInstance(IOVSyncValue{Timestamp(5)});
      EventSetup eventSetup{provider.eventSetupImpl(), 0, nullptr, pc};
      REQUIRE(3 == eventSetup.get<DummyRecord>().cacheIdentifier());
    }
  }

  SECTION("resetResolversTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<DummyFinder> finder = std::make_shared<DummyFinder>();
    finder->setInterval(ValidityInterval(IOVSyncValue(Timestamp(2)), IOVSyncValue(Timestamp(3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(finder));

    ComponentDescription description("DummyESProductResolverProvider", "", 0, true);
    ParameterSet ps;
    ps.addParameter<std::string>("name", "test11");
    ps.registerIt();
    description.pid_ = ps.id();
    DummyData kOne{1};
    auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kOne);
    dummyProv->setDescription(description);
    provider.add(dummyProv);

    edm::ESParentContext pc;
    {
      controller.eventSetupForInstance(IOVSyncValue{Timestamp(2)});
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      REQUIRE(2 == eventSetup.get<DummyRecord>().cacheIdentifier());
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(data->value_ == 1);
    }
    provider.forceCacheClear();
    {
      controller.eventSetupForInstance(IOVSyncValue{Timestamp(2)});
      DummyDataConsumer consumer{edm::ESInputTag("", "")};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      EventSetup eventSetup{provider.eventSetupImpl(),
                            static_cast<unsigned int>(edm::Transition::Event),
                            consumer.esGetTokenIndices(edm::Transition::Event),
                            pc};
      eventSetup.get<DummyRecord>();
      REQUIRE(3 == eventSetup.get<DummyRecord>().cacheIdentifier());
      dummyProv->incrementData();
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(data->value_ == 2);
    }
  }
}
