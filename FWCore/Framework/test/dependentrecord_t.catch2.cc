/*
 *  dependentrecord_t.cpp
 *  EDMProto
 *
 *  Created by Chris Jones on 4/29/05.
 *  Changed by Viji Sundararajan on 29-Jun-2005
 *
 */

#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/Dummy2Record.h"
#include "FWCore/Framework/test/DepRecord.h"
#include "FWCore/Framework/test/DepOnDepRecord.h"
#include "FWCore/Framework/test/DepOn2Record.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyESProductResolverProvider.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/SupportingRecordIntervalFinderHelper.h"
#include "FWCore/Framework/src/SynchronousEventSetupsController.h"
#include "FWCore/Framework/src/makeFindersForRecords.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/test/print_eventsetup_record_dependencies.h"
#include "FWCore/Framework/interface/ESModuleProducesInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "catch2/catch_all.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstring>

using namespace edm::eventsetup;

namespace {
  edm::ParameterSet createDummyPset() {
    edm::ParameterSet pset;
    std::vector<std::string> emptyVStrings;
    pset.addParameter<std::vector<std::string>>("@all_esprefers", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_essources", emptyVStrings);
    pset.addParameter<std::vector<std::string>>("@all_esmodules", emptyVStrings);
    return pset;
  }
}  // namespace

/* The Records used in the test have the following dependencies
   DepOnDepRecord -----> DepRecord
                
   DepRecord -----> DummyRecord
                /
   DepOn2Record---> Dummy2Record

 */

namespace {

  edm::ActivityRegistry activityRegistry;

  class DummyESProductResolverProvider : public edm::eventsetup::ESProductResolverProvider {
  public:
    DummyESProductResolverProvider() { usingRecord<DummyRecord>(); }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedResolversVector();
    }
    std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
      return std::vector<edm::eventsetup::ESModuleProducesInfo>();
    }
  };

  class DepRecordResolverProvider : public edm::eventsetup::ESProductResolverProvider {
  public:
    DepRecordResolverProvider() { usingRecord<DepRecord>(); }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedResolversVector();
    }
    std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
      return std::vector<edm::eventsetup::ESModuleProducesInfo>();
    }
  };

  class DepOnDepRecordResolverProvider : public edm::eventsetup::ESProductResolverProvider {
  public:
    DepOnDepRecordResolverProvider() { usingRecord<DepOnDepRecord>(); }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedResolversVector();
    }
    std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
      return std::vector<edm::eventsetup::ESModuleProducesInfo>();
    }
  };

  class WorkingDepRecordResolver
      : public edm::eventsetup::ESProductResolverTemplate<DepRecord, edm::eventsetup::test::DummyData> {
  public:
    WorkingDepRecordResolver(const edm::eventsetup::test::DummyData* iDummy) : data_(iDummy) {}

  protected:
    const value_type* make(const record_type&, const DataKey&) final { return data_; }
    void const* getAfterPrefetchImpl() const final { return data_; }

  private:
    const edm::eventsetup::test::DummyData* data_;
  };

  class DepRecordResolverProviderWithData : public edm::eventsetup::ESProductResolverProvider {
  public:
    DepRecordResolverProviderWithData(const edm::eventsetup::test::DummyData& iData = edm::eventsetup::test::DummyData())
        : dummy_(iData) {
      usingRecord<DepRecord>();
    }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      KeyedResolversVector keyedResolversVector;
      std::shared_ptr<WorkingDepRecordResolver> pResolver = std::make_shared<WorkingDepRecordResolver>(&dummy_);
      edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<edm::eventsetup::test::DummyData>(), "");
      keyedResolversVector.emplace_back(dataKey, pResolver);
      return keyedResolversVector;
    }
    std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
      return std::vector<edm::eventsetup::ESModuleProducesInfo>(
          1,
          edm::eventsetup::ESModuleProducesInfo(
              edm::eventsetup::EventSetupRecordKey::makeKey<DepRecord>(),
              edm::eventsetup::DataKey(edm::eventsetup::DataKey::makeTypeTag<edm::eventsetup::test::DummyData>(), ""),
              0));
    }

  private:
    edm::eventsetup::test::DummyData dummy_;
  };

  class DepOn2RecordResolverProvider : public edm::eventsetup::ESProductResolverProvider {
  public:
    DepOn2RecordResolverProvider() { usingRecord<DepOn2Record>(); }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedResolversVector();
    }
    std::vector<edm::eventsetup::ESModuleProducesInfo> producesInfo() const override {
      return std::vector<edm::eventsetup::ESModuleProducesInfo>();
    }
  };

  template <typename RECORD>
  class TestFinder : public edm::EventSetupRecordIntervalFinder {
  public:
    TestFinder() : edm::EventSetupRecordIntervalFinder(), interval_() { this->findingRecord<RECORD>(); }

    void setInterval(const edm::ValidityInterval& iInterval) {
      interval_ = iInterval;
      resetInterval(key());
    }

    edm::eventsetup::EventSetupRecordKey key() const { return edm::eventsetup::EventSetupRecordKey::makeKey<RECORD>(); }

  protected:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                        const edm::IOVSyncValue& iTime,
                        edm::ValidityInterval& iInterval) override {
      if (interval_.validFor(iTime)) {
        iInterval = interval_;
      } else {
        if (interval_.last() == edm::IOVSyncValue::invalidIOVSyncValue() &&
            interval_.first() != edm::IOVSyncValue::invalidIOVSyncValue() && interval_.first() <= iTime) {
          iInterval = interval_;
        } else {
          iInterval = edm::ValidityInterval();
        }
      }
    }

  private:
    edm::ValidityInterval interval_;
  };

  using DepRecordFinder = TestFinder<DepRecord>;
  using DummyRecordFinder = TestFinder<DummyRecord>;
  using Dummy2RecordFinder = TestFinder<Dummy2Record>;
  template <typename RECORD>
  struct DummyDataConsumer : public edm::EDConsumerBase {
    explicit DummyDataConsumer(edm::ESInputTag const& iTag)
        : m_token{esConsumes<edm::eventsetup::test::DummyData, RECORD>(iTag)} {}

    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& resolvers = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != resolvers.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          edm::ServiceToken token;
          rec->prefetchAsync(
              edm::WaitingTaskHolder(group, &waitTask), resolvers[i], &iImpl, token, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }
    edm::ESGetToken<edm::eventsetup::test::DummyData, RECORD> m_token;
  };

  template <typename RECORD1, typename RECORD2>
  struct DummyDataConsumer2 : public edm::EDConsumerBase {
    explicit DummyDataConsumer2(edm::ESInputTag const& iTag1, edm::ESInputTag const& iTag2)
        : m_token1{esConsumes<edm::eventsetup::test::DummyData, RECORD1>(iTag1)},
          m_token2{esConsumes<edm::eventsetup::test::DummyData, RECORD2>(iTag2)} {}

    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& resolvers = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != resolvers.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          edm::ServiceToken token;
          rec->prefetchAsync(
              edm::WaitingTaskHolder(group, &waitTask), resolvers[i], &iImpl, token, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }

    edm::ESGetToken<edm::eventsetup::test::DummyData, RECORD1> m_token1;
    edm::ESGetToken<edm::eventsetup::test::DummyData, RECORD2> m_token2;
  };
}  // namespace

using namespace edm::eventsetup;
TEST_CASE("DependentRecord", "[Framework][EventSetup]") {
  auto m_scheduler = std::make_unique<edm::ThreadsController>(1);

  SECTION("dependentConstructorTest") {
    EventSetupRecordProvider depProvider(DepRecord::keyForClass(), &activityRegistry);

    REQUIRE(1 == depProvider.supportingRecords().size());
    REQUIRE(*(depProvider.supportingRecords().begin()) == DummyRecord::keyForClass());

    edm::print_eventsetup_record_dependencies<DepRecord>(std::cout);
  }

  SECTION("dependentFinder1Test") {
    const edm::EventID eID_1(1, 1, 1);
    const edm::IOVSyncValue sync_1(eID_1);
    const edm::EventID eID_3(1, 1, 3);
    const edm::ValidityInterval definedInterval(sync_1, edm::IOVSyncValue(eID_3));
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(definedInterval);

    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);
    finder.addSupporter(SupportingRecordIntervalFinderHelper(DummyRecord::keyForClass(), dummyFinder));

    REQUIRE(definedInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2))));

    dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
    REQUIRE(edm::ValidityInterval::invalidInterval() ==
            finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));

    const edm::EventID eID_5(1, 1, 5);
    const edm::IOVSyncValue sync_5(eID_5);
    const edm::ValidityInterval unknownedEndInterval(sync_5, edm::IOVSyncValue::invalidIOVSyncValue());
    dummyFinder->setInterval(unknownedEndInterval);

    REQUIRE(unknownedEndInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));
  }

  SECTION("dependentFinder2Test") {
    auto dummyFinder1 = std::make_shared<DummyRecordFinder>();
    auto dummyFinder2 = std::make_shared<Dummy2RecordFinder>();

    const edm::EventID eID_1(1, 1, 1);
    const edm::IOVSyncValue sync_1(eID_1);
    const edm::ValidityInterval definedInterval1(sync_1, edm::IOVSyncValue(edm::EventID(1, 1, 5)));
    dummyFinder1->setInterval(definedInterval1);

    auto dummyProvider2 = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

    const edm::EventID eID_2(1, 1, 2);
    const edm::IOVSyncValue sync_2(eID_2);
    const edm::ValidityInterval definedInterval2(sync_2, edm::IOVSyncValue(edm::EventID(1, 1, 6)));
    dummyFinder2->setInterval(definedInterval2);

    const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                                std::min(definedInterval1.last(), definedInterval2.last()));

    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();

    DependentRecordIntervalFinder finder(depRecordKey);
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinder1->key(), dummyFinder1));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinder2->key(), dummyFinder2));

    REQUIRE(overlapInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));
  }

  SECTION("testIncomparibleIOVAlgorithm") {
    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);

    //test case where we have two providers, one synching on time the other on run/lumi/event
    auto dummyFinderEventID = std::make_shared<DummyRecordFinder>();
    auto dummyFinderTime = std::make_shared<DummyRecordFinder>();

    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderEventID->key(), dummyFinderEventID));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderTime->key(), dummyFinderTime));

    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 5)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(6)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      // The algorithm selects incomparable IOVs from the two providers based on
      // the least estimated time difference from the start of changed IOVs.
      // The difference based on lumi numbers is zero in this case (assumes
      // 23 second lumis). The difference in times is also zero because it is
      // calculating in seconds and the lower 32 bits of the time argument to
      // the Timestamp constructor are dropped by the algorithm. So when finding
      // the closest both time and eventID estimates are zero. On a tie, the algorithm
      // selects the time IOV.
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(3))));

      // With IOVs that are not comparable we just continually get back the same
      // result when none of the IOVs changes. Next 3 just show that.

      //should give back same interval
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(4))));

      //should give back same interval
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(3))));

      //should give back same interval
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2))));
    }
    {
      //Change only run/lumi/event based provider. The algorithm picks
      //the closest of the IOVs that changed, so now gets the EventID based
      //IOV.
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 10)));
      dummyFinderEventID->setInterval(iovEventID);

      REQUIRE(iovEventID ==
              dummyFinderEventID->findIntervalFor(dummyFinderEventID->key(),
                                                  edm::IOVSyncValue(edm::EventID(1, 1, 6), edm::Timestamp(5))));
      const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 6), edm::Timestamp(5))));
    }
    {
      //Change only time based provider
      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
    }
    //Change both but make run/lumi/event 'closer' by having same lumi
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 2, 11)),
                                             edm::IOVSyncValue(edm::EventID(1, 3, 20)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1ULL << 32)),
                                          edm::IOVSyncValue(edm::Timestamp(5ULL << 32)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV == finder.findIntervalFor(
                                 depRecordKey, edm::IOVSyncValue(edm::EventID(1, 2, 12), edm::Timestamp(3ULL << 32))));
    }
    //Change both but make time 'closer'
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 3, 21)),
                                             edm::IOVSyncValue(edm::EventID(1, 10, 40)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7ULL << 32)),
                                          edm::IOVSyncValue(edm::Timestamp(10ULL << 32)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV == finder.findIntervalFor(
                                 depRecordKey, edm::IOVSyncValue(edm::EventID(1, 4, 30), edm::Timestamp(8ULL << 32))));
    }
    //Change both but make run/lumi/event 'closer'
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 11, 41)),
                                             edm::IOVSyncValue(edm::EventID(1, 20, 60)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(11ULL << 32)),
                                          edm::IOVSyncValue(edm::Timestamp(100ULL << 32)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey,
                                     edm::IOVSyncValue(edm::EventID(1, 12, 50), edm::Timestamp(70ULL << 32))));
    }

    //Change both and make it ambiguous because of different run #
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(2, 1, 0)),
                                             edm::IOVSyncValue(edm::EventID(6, 0, 0)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(200ULL << 32)),
                                          edm::IOVSyncValue(edm::Timestamp(500ULL << 32)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey,
                                     edm::IOVSyncValue(edm::EventID(4, 12, 50), edm::Timestamp(400ULL << 32))));
    }
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 10)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
    }
  }

  SECTION("testInvalidIOVAlgorithm") {
    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);

    auto dummyFinderEventID = std::make_shared<DummyRecordFinder>();
    auto dummyFinderTime = std::make_shared<DummyRecordFinder>();

    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderEventID->key(), dummyFinderEventID));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderTime->key(), dummyFinderTime));

    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 10)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
    }

    //check with invalid intervals
    const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                        edm::IOVSyncValue::invalidIOVSyncValue());
    {
      dummyFinderEventID->setInterval(invalid);
      const edm::ValidityInterval expectedIOV(edm::IOVSyncValue(edm::Timestamp(7)),
                                              edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 11), edm::Timestamp(8))));
    }
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 12)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 20)));
      dummyFinderEventID->setInterval(iovEventID);
      dummyFinderTime->setInterval(invalid);
      const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 13), edm::Timestamp(11))));
    }
    {
      dummyFinderEventID->setInterval(invalid);
      dummyFinderTime->setInterval(invalid);

      REQUIRE(invalid ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 13), edm::Timestamp(11))));
    }
  }

  SECTION("testInvalidIOVFirstTime") {
    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);

    auto dummyFinderEventID = std::make_shared<DummyRecordFinder>();
    auto dummyFinderTime = std::make_shared<DummyRecordFinder>();

    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderEventID->key(), dummyFinderEventID));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderTime->key(), dummyFinderTime));

    const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                        edm::IOVSyncValue::invalidIOVSyncValue());
    {
      //check for bug which only happens the first time we synchronize
      // have the second one invalid
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 6)));
      dummyFinderEventID->setInterval(iovEventID);

      dummyFinderTime->setInterval(invalid);

      const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(1))));
    }
    {
      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(2)), edm::IOVSyncValue(edm::Timestamp(6)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(3))));
    }
  }

  SECTION("testInvalidIOVFirstEventID") {
    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);

    auto dummyFinderEventID = std::make_shared<DummyRecordFinder>();
    auto dummyFinderTime = std::make_shared<DummyRecordFinder>();

    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderEventID->key(), dummyFinderEventID));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinderTime->key(), dummyFinderTime));

    const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                        edm::IOVSyncValue::invalidIOVSyncValue());
    {
      //check for bug which only happens the first time we synchronize
      // have the  first one invalid
      dummyFinderEventID->setInterval(invalid);

      const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(6)));
      dummyFinderTime->setInterval(iovTime);

      const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(3))));
    }
    {
      const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 5)),
                                             edm::IOVSyncValue(edm::EventID(1, 1, 10)));
      dummyFinderEventID->setInterval(iovEventID);

      const edm::ValidityInterval expectedIOV(edm::IOVSyncValue(edm::Timestamp(1)),
                                              edm::IOVSyncValue::invalidIOVSyncValue());
      REQUIRE(expectedIOV ==
              finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(4))));
    }
  }

  SECTION("timeAndRunTest") {
    edm::ParameterSet pset = createDummyPset();
    {
      SynchronousEventSetupsController controller;
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

      std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
          std::make_shared<DummyESProductResolverProvider>();
      controller.addExtra(dummyProv);

      std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
      dummyFinder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 5))));
      controller.addExtra(dummyFinder);

      std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv =
          std::make_shared<DepOn2RecordResolverProvider>();
      controller.addExtra(depProv);

      std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
      dummy2Finder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(5))));
      controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));

      {
        edm::ESParentContext parentC;
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
        const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2)));
        const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id2 = eventSetup2.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id2);

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(2)));
        const edm::EventSetup eventSetup3(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id3 = eventSetup3.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id3);

        dummy2Finder->setInterval(
            edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(6)), edm::IOVSyncValue(edm::Timestamp(10))));

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(7)));
        const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id4 = eventSetup4.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 != id4);

        dummyFinder->setInterval(
            edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 6)), edm::IOVSyncValue(edm::EventID(1, 1, 10))));

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(8)));
        const edm::EventSetup eventSetup5(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id5 = eventSetup5.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id4 != id5);
      }
    }

    {
      //check that going all the way through EventSetup works properly
      // using two records with open ended IOVs
      SynchronousEventSetupsController controller;
      EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

      std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
          std::make_shared<DummyESProductResolverProvider>();
      controller.addExtra(dummyProv);

      std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
      dummyFinder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue::invalidIOVSyncValue()));
      controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

      std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv =
          std::make_shared<DepOn2RecordResolverProvider>();
      controller.addExtra(depProv);

      std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
      dummy2Finder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue::invalidIOVSyncValue()));
      controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));
      {
        edm::ESParentContext parentC;
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
        const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2)));
        const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id2 = eventSetup2.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id2);

        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(2)));
        const edm::EventSetup eventSetup3(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id3 = eventSetup3.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id3);

        dummy2Finder->setInterval(
            edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(6)), edm::IOVSyncValue::invalidIOVSyncValue()));
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(7)));
        const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id4 = eventSetup4.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 != id4);

        dummyFinder->setInterval(
            edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 6)), edm::IOVSyncValue::invalidIOVSyncValue()));
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(8)));
        const edm::EventSetup eventSetup5(provider.eventSetupImpl(), 0, nullptr, parentC);
        long long id5 = eventSetup5.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id4 != id5);
      }
    }
  }

  SECTION("dependentSetFindersTest") {
    auto depProvider = std::make_unique<EventSetupRecordProvider>(DepRecord::keyForClass(), &activityRegistry);

    auto dummyProvider = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    std::vector<std::shared_ptr<edm::EventSetupRecordIntervalFinder>> finders;
    finders.push_back(dummyFinder);

    std::set<EventSetupRecordKey> neededRecords;
    neededRecords.insert(dummyProvider->key());
    neededRecords.insert(depProvider->key());
    {
      auto supporters = depProvider->supportingRecords();
      neededRecords.insert(supporters.begin(), supporters.end());
    }

    REQUIRE(*(depProvider->supportingRecords().begin()) == dummyProvider->key());

    std::map<EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>> keyToFinders =
        edm::impl::makeFindersForRecords(neededRecords, finders);

    auto interval =
        keyToFinders[depProvider->key()]->findIntervalFor(depProvider->key(), edm::IOVSyncValue(edm::EventID(1, 1, 2)));
    REQUIRE(interval.first() == edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    REQUIRE(interval.last() == edm::IOVSyncValue(edm::EventID(1, 1, 3)));
  }

  SECTION("indirect dependencies") {
    std::vector<std::shared_ptr<edm::EventSetupRecordIntervalFinder>> finders;
    std::set<EventSetupRecordKey> neededRecords;

    auto depProvider = std::make_unique<EventSetupRecordProvider>(DepRecord::keyForClass(), &activityRegistry);
    neededRecords.insert(depProvider->key());

    auto depOnDepProvider =
        std::make_unique<EventSetupRecordProvider>(DepOnDepRecord::keyForClass(), &activityRegistry);
    neededRecords.insert(depOnDepProvider->key());

    auto dummyProvider = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
    neededRecords.insert(dummyProvider->key());

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    finders.push_back(dummyFinder);

    REQUIRE(*(depProvider->supportingRecords().begin()) == dummyProvider->key());

    std::map<EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>> keyToFinders =
        edm::impl::makeFindersForRecords(neededRecords, finders);

    auto interval =
        keyToFinders[depProvider->key()]->findIntervalFor(depProvider->key(), edm::IOVSyncValue(edm::EventID(1, 1, 2)));
    REQUIRE(interval.first() == edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    REQUIRE(interval.last() == edm::IOVSyncValue(edm::EventID(1, 1, 3)));

    interval = keyToFinders[depOnDepProvider->key()]->findIntervalFor(depOnDepProvider->key(),
                                                                      edm::IOVSyncValue(edm::EventID(1, 1, 2)));
    REQUIRE(interval.first() == edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    REQUIRE(interval.last() == edm::IOVSyncValue(edm::EventID(1, 1, 3)));
  }

  SECTION("getTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
        std::make_shared<DummyESProductResolverProvider>();
    controller.addExtra(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv = std::make_shared<DepRecordResolverProvider>();
    controller.addExtra(depProv);
    edm::ESParentContext parentC;
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
      const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
      const DepRecord& depRecord = eventSetup1.get<DepRecord>();

      depRecord.getRecord<DummyRecord>();

      auto dr = depRecord.tryToGetRecord<DummyRecord>();
      REQUIRE(dr.has_value());
    }
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4)));
      const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
      REQUIRE_THROWS_AS(eventSetup2.get<DepRecord>(), edm::eventsetup::NoRecordException<DepRecord>);
    }
  }

  SECTION("getDataWithESGetTokenTest") {
    using edm::eventsetup::test::DummyData;
    using edm::eventsetup::test::DummyESProductResolverProvider;
    DummyData kGood{1};
    DummyData kBad{0};

    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    {
      edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
      dummyProv->setDescription(description);
      controller.addExtra(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DepRecordResolverProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordResolverProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      controller.addExtra(dummyProv);
    }

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& data = eventSetup.getData(consumer.m_token);
      REQUIRE(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& data = eventSetup.getData(consumer.m_token);
      REQUIRE(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();
      auto const& data = depRecord.get(consumer.m_token);
      REQUIRE(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data = depRecord.get(consumer.m_token);
      REQUIRE(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data = depRecord.get(consumer.m_token);
      REQUIRE(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data1 = depRecord.get(consumer.m_token1);
      REQUIRE(kBad.value_ == data1.value_);

      auto const& data2 = depRecord.get(consumer.m_token2);
      REQUIRE(kGood.value_ == data2.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      REQUIRE_THROWS_AS(depRecord.get(consumer.m_token), cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      REQUIRE_THROWS_AS(depRecord.get(consumer.m_token), cms::Exception);
    }
  }

  SECTION("getHandleWithESGetTokenTest") {
    using edm::eventsetup::test::DummyData;
    using edm::eventsetup::test::DummyESProductResolverProvider;
    DummyData kGood{1};
    DummyData kBad{0};

    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    {
      edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
      dummyProv->setDescription(description);
      controller.addExtra(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DepRecordResolverProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordResolverProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      controller.addExtra(dummyProv);
    }
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));

    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      REQUIRE(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      REQUIRE(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data1 = depRecord.getHandle(consumer.m_token1);
      REQUIRE(kBad.value_ == data1->value_);
      const edm::eventsetup::ComponentDescription* desc = data1.description();
      REQUIRE(desc->label_ == "testOne");

      edm::ESHandle<DummyData> data2 = depRecord.getHandle(consumer.m_token2);
      REQUIRE(kGood.value_ == data2->value_);
      desc = data2.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      REQUIRE(not depRecord.getHandle(consumer.m_token));
      REQUIRE_THROWS_AS(*depRecord.getHandle(consumer.m_token), cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      REQUIRE(not depRecord.getHandle(consumer.m_token));
      REQUIRE_THROWS_AS(*depRecord.getHandle(consumer.m_token), cms::Exception);
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

    {
      edm::eventsetup::ComponentDescription description("DummyESProductResolverProvider", "testOne", 0, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyESProductResolverProvider>(kBad);
      dummyProv->setDescription(description);
      controller.addExtra(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DepRecordResolverProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordResolverProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      controller.addExtra(dummyProv);
    }
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      REQUIRE(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      REQUIRE(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      REQUIRE(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data1 = depRecord.getTransientHandle(consumer.m_token1);
      REQUIRE(kBad.value_ == data1->value_);
      const edm::eventsetup::ComponentDescription* desc = data1.description();
      REQUIRE(desc->label_ == "testOne");

      edm::ESTransientHandle<DummyData> data2 = depRecord.getTransientHandle(consumer.m_token2);
      REQUIRE(kGood.value_ == data2->value_);
      desc = data2.description();
      REQUIRE(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      REQUIRE(not data);
      REQUIRE_THROWS_AS(*data, cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToResolverIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      REQUIRE(not data);
      REQUIRE_THROWS_AS(*data, cms::Exception);
    }
  }

  SECTION("oneOfTwoRecordTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
        std::make_shared<DummyESProductResolverProvider>();
    controller.addExtra(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv =
        std::make_shared<DepOn2RecordResolverProvider>();
    controller.addExtra(depProv);
    {
      edm::ESParentContext parentC;
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
      const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
      const DepOn2Record& depRecord = eventSetup1.get<DepOn2Record>();

      depRecord.getRecord<DummyRecord>();
      REQUIRE_THROWS_AS(depRecord.getRecord<Dummy2Record>(), edm::eventsetup::NoRecordException<Dummy2Record>);

      try {
        depRecord.getRecord<Dummy2Record>();
      } catch (edm::eventsetup::NoRecordException<Dummy2Record>& e) {
        //make sure that the record name appears in the error message.
        REQUIRE(0 != strstr(e.what(), "DepOn2Record"));
        REQUIRE(0 != strstr(e.what(), "Dummy2Record"));
        // std::cout<<e.what()<<std::endl;
      }
    }
  }

  SECTION("resetTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    auto provider = controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
        std::make_shared<DummyESProductResolverProvider>();
    controller.addExtra(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv = std::make_shared<DepRecordResolverProvider>();
    controller.addExtra(depProv);
    {
      edm::ESParentContext parentC;
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
      const edm::EventSetup eventSetup1(provider->eventSetupImpl(), 0, nullptr, parentC);
      const DepRecord& depRecord = eventSetup1.get<DepRecord>();
      unsigned long long depCacheID = depRecord.cacheIdentifier();
      const DummyRecord& dummyRecord = depRecord.getRecord<DummyRecord>();
      unsigned long long dummyCacheID = dummyRecord.cacheIdentifier();

      controller.resetRecordPlusDependentRecords(dummyRecord.key());
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
      REQUIRE(dummyCacheID != dummyRecord.cacheIdentifier());
      REQUIRE(depCacheID != depRecord.cacheIdentifier());
    }
  }

  SECTION("alternateFinderTest") {
    const edm::EventID eID_1(1, 1, 1);
    const edm::IOVSyncValue sync_1(eID_1);
    const edm::EventID eID_3(1, 1, 3);
    const edm::IOVSyncValue sync_3(eID_3);
    const edm::EventID eID_4(1, 1, 4);
    const edm::ValidityInterval definedInterval(sync_1, edm::IOVSyncValue(eID_4));
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(definedInterval);

    std::shared_ptr<DepRecordFinder> depFinder = std::make_shared<DepRecordFinder>();
    const edm::EventID eID_2(1, 1, 2);
    const edm::IOVSyncValue sync_2(eID_2);
    const edm::ValidityInterval depInterval(sync_1, sync_2);
    depFinder->setInterval(depInterval);

    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);
    finder.setAlternateFinder(depFinder);
    finder.addSupporter(SupportingRecordIntervalFinderHelper(DummyRecord::keyForClass(), dummyFinder));

    REQUIRE(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1))));

    const edm::ValidityInterval dep2Interval(sync_3, edm::IOVSyncValue(eID_4));
    depFinder->setInterval(dep2Interval);

    /*const edm::ValidityInterval tempIOV = */
    finder.findIntervalFor(depRecordKey, sync_3);
    //std::cout <<  tempIOV.first().eventID()<<" to "<<tempIOV.last().eventID() <<std::endl;
    REQUIRE(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3));

    dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
    depFinder->setInterval(edm::ValidityInterval::invalidInterval());

    REQUIRE(edm::ValidityInterval::invalidInterval() ==
            finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));

    const edm::EventID eID_6(1, 1, 6);
    const edm::IOVSyncValue sync_6(eID_6);
    const edm::ValidityInterval unknownedEndInterval(sync_6, edm::IOVSyncValue::invalidIOVSyncValue());
    dummyFinder->setInterval(unknownedEndInterval);

    const edm::EventID eID_7(1, 1, 7);
    const edm::IOVSyncValue sync_7(eID_7);
    const edm::ValidityInterval iov6_7(sync_6, sync_7);
    depFinder->setInterval(iov6_7);

    REQUIRE(unknownedEndInterval == finder.findIntervalFor(depRecordKey, sync_6));

    //see if dependent record can override the finder
    dummyFinder->setInterval(depInterval);
    depFinder->setInterval(definedInterval);
    REQUIRE(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1))));

    dummyFinder->setInterval(dep2Interval);
    REQUIRE(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3));
  }

  SECTION("invalidRecordTest") {
    auto dummyFinder1 = std::make_shared<DummyRecordFinder>();

    const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                        edm::IOVSyncValue::invalidIOVSyncValue());

    dummyFinder1->setInterval(invalid);

    auto dummyFinder2 = std::make_shared<DummyRecordFinder>();
    dummyFinder2->setInterval(invalid);

    const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
    DependentRecordIntervalFinder finder(depRecordKey);
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinder1->key(), dummyFinder1));
    finder.addSupporter(SupportingRecordIntervalFinderHelper(dummyFinder2->key(), dummyFinder2));

    REQUIRE(invalid == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2))));

    const edm::EventID eID_1(1, 1, 5);
    const edm::IOVSyncValue sync_1(eID_1);
    const edm::ValidityInterval definedInterval1(sync_1, edm::IOVSyncValue(edm::EventID(1, 1, 10)));
    const edm::EventID eID_2(1, 1, 2);
    const edm::IOVSyncValue sync_2(eID_2);
    const edm::ValidityInterval definedInterval2(sync_2, edm::IOVSyncValue(edm::EventID(1, 1, 6)));
    dummyFinder2->setInterval(definedInterval2);

    const edm::ValidityInterval openEnded1(definedInterval2.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    REQUIRE(openEnded1 == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));

    dummyFinder1->setInterval(definedInterval1);

    const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                                std::min(definedInterval1.last(), definedInterval2.last()));

    REQUIRE(overlapInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));

    dummyFinder2->setInterval(invalid);
    const edm::ValidityInterval openEnded2(definedInterval1.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    REQUIRE(openEnded2 == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7))));
  }

  SECTION("extendIOVTest") {
    SynchronousEventSetupsController controller;
    edm::ParameterSet pset = createDummyPset();
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> dummyProv =
        std::make_shared<DummyESProductResolverProvider>();
    controller.addExtra(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

    edm::IOVSyncValue startSyncValue{edm::EventID{1, 1, 1}};
    dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 5}}});
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>{dummyFinder});

    std::shared_ptr<edm::eventsetup::ESProductResolverProvider> depProv =
        std::make_shared<DepOn2RecordResolverProvider>();
    controller.addExtra(depProv);

    edm::ESParentContext parentC;
    std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
    dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 6}}});
    controller.addExtra(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
      const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();
      REQUIRE(id1 == eventSetup1.get<DummyRecord>().cacheIdentifier());
      REQUIRE(id1 == eventSetup1.get<Dummy2Record>().cacheIdentifier());

      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(2)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id);
        REQUIRE(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }
      //extend the IOV DummyRecord while Dummy2Record still covers this range
      dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 7}}});
      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 6), edm::Timestamp(7)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id);
        REQUIRE(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }

      //extend the IOV Dummy2Record while DummyRecord still covers this range
      dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 7}}});

      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id);
        REQUIRE(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }
      //extend the both IOVs
      dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 8}}});

      dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 8}}});
      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 8), edm::Timestamp(7)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 == id);
        REQUIRE(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }
      //extend only one and create a new IOV for the other
      dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 9}}});

      dummyFinder->setInterval(
          edm::ValidityInterval{edm::IOVSyncValue{edm::EventID{1, 1, 9}}, edm::IOVSyncValue{edm::EventID{1, 1, 9}}});
      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 9), edm::Timestamp(7)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 + 1 == id);
        REQUIRE(id1 + 1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }
      //extend the otherone and create a new IOV for the other
      dummy2Finder->setInterval(
          edm::ValidityInterval{edm::IOVSyncValue{edm::EventID{1, 1, 10}}, edm::IOVSyncValue{edm::EventID{1, 1, 10}}});

      dummyFinder->setInterval(
          edm::ValidityInterval{edm::IOVSyncValue{edm::EventID{1, 1, 9}}, edm::IOVSyncValue{edm::EventID{1, 1, 10}}});

      {
        controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 10), edm::Timestamp(7)));
        const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
        unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
        REQUIRE(id1 + 2 == id);
        REQUIRE(id1 + 1 == eventSetup.get<DummyRecord>().cacheIdentifier());
        REQUIRE(id1 + 1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
      }
    }
  }
}
