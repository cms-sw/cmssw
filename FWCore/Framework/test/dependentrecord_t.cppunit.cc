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
#include "FWCore/Framework/test/DepOn2Record.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/src/SynchronousEventSetupsController.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/test/print_eventsetup_record_dependencies.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "cppunit/extensions/HelperMacros.h"

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

class testdependentrecord : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testdependentrecord);

  CPPUNIT_TEST(dependentConstructorTest);
  CPPUNIT_TEST(dependentFinder1Test);
  CPPUNIT_TEST(dependentFinder2Test);
  CPPUNIT_TEST(testIncomparibleIOVAlgorithm);
  CPPUNIT_TEST(testInvalidIOVAlgorithm);
  CPPUNIT_TEST(testInvalidIOVFirstTime);
  CPPUNIT_TEST(testInvalidIOVFirstEventID);
  CPPUNIT_TEST(timeAndRunTest);
  CPPUNIT_TEST(dependentSetproviderTest);
  CPPUNIT_TEST(getTest);
  CPPUNIT_TEST(getDataWithESGetTokenTest);
  CPPUNIT_TEST(getHandleWithESGetTokenTest);
  CPPUNIT_TEST(getTransientHandleWithESGetTokenTest);
  CPPUNIT_TEST(oneOfTwoRecordTest);
  CPPUNIT_TEST(resetTest);
  CPPUNIT_TEST(alternateFinderTest);
  CPPUNIT_TEST(invalidRecordTest);
  CPPUNIT_TEST(extendIOVTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() { m_scheduler = std::make_unique<edm::ThreadsController>(1); }
  void tearDown() {}

  void dependentConstructorTest();
  void dependentFinder1Test();
  void dependentFinder2Test();
  void testIncomparibleIOVAlgorithm();
  void testInvalidIOVAlgorithm();
  void testInvalidIOVFirstTime();
  void testInvalidIOVFirstEventID();
  void timeAndRunTest();
  void dependentSetproviderTest();
  void getTest();
  void getDataWithESGetTokenTest();
  void getHandleWithESGetTokenTest();
  void getTransientHandleWithESGetTokenTest();
  void oneOfTwoRecordTest();
  void resetTest();
  void alternateFinderTest();
  void invalidRecordTest();
  void extendIOVTest();

private:
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;

};  //Cppunit class declaration over

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testdependentrecord);

/* The Records used in the test have the following dependencies
   DepRecord -----> DummyRecord
                /
   DepOn2Record---> Dummy2Record

 */

namespace {

  edm::ActivityRegistry activityRegistry;

  class DummyProxyProvider : public edm::eventsetup::DataProxyProvider {
  public:
    DummyProxyProvider() { usingRecord<DummyRecord>(); }

  protected:
    KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedProxiesVector();
    }
  };

  class DepRecordProxyProvider : public edm::eventsetup::DataProxyProvider {
  public:
    DepRecordProxyProvider() { usingRecord<DepRecord>(); }

  protected:
    KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedProxiesVector();
    }
  };

  class WorkingDepRecordProxy : public edm::eventsetup::DataProxyTemplate<DepRecord, edm::eventsetup::test::DummyData> {
  public:
    WorkingDepRecordProxy(const edm::eventsetup::test::DummyData* iDummy) : data_(iDummy) {}

  protected:
    const value_type* make(const record_type&, const DataKey&) final { return data_; }
    void const* getAfterPrefetchImpl() const final { return data_; }

  private:
    const edm::eventsetup::test::DummyData* data_;
  };

  class DepRecordProxyProviderWithData : public edm::eventsetup::DataProxyProvider {
  public:
    DepRecordProxyProviderWithData(const edm::eventsetup::test::DummyData& iData = edm::eventsetup::test::DummyData())
        : dummy_(iData) {
      usingRecord<DepRecord>();
    }

  protected:
    KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      KeyedProxiesVector keyedProxiesVector;
      std::shared_ptr<WorkingDepRecordProxy> pProxy = std::make_shared<WorkingDepRecordProxy>(&dummy_);
      edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<edm::eventsetup::test::DummyData>(), "");
      keyedProxiesVector.emplace_back(dataKey, pProxy);
      return keyedProxiesVector;
    }

  private:
    edm::eventsetup::test::DummyData dummy_;
  };

  class DepOn2RecordProxyProvider : public edm::eventsetup::DataProxyProvider {
  public:
    DepOn2RecordProxyProvider() { usingRecord<DepOn2Record>(); }

  protected:
    KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      return KeyedProxiesVector();
    }
  };

  class DepRecordFinder : public edm::EventSetupRecordIntervalFinder {
  public:
    DepRecordFinder() : edm::EventSetupRecordIntervalFinder(), interval_() { this->findingRecord<DepRecord>(); }

    void setInterval(const edm::ValidityInterval& iInterval) { interval_ = iInterval; }

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

  class Dummy2RecordFinder : public edm::EventSetupRecordIntervalFinder {
  public:
    Dummy2RecordFinder() : edm::EventSetupRecordIntervalFinder(), interval_() { this->findingRecord<Dummy2Record>(); }

    void setInterval(const edm::ValidityInterval& iInterval) { interval_ = iInterval; }

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
}  // namespace

using namespace edm::eventsetup;
void testdependentrecord::dependentConstructorTest() {
  EventSetupRecordProvider depProvider(DepRecord::keyForClass(), &activityRegistry);

  CPPUNIT_ASSERT(1 == depProvider.dependentRecords().size());
  CPPUNIT_ASSERT(*(depProvider.dependentRecords().begin()) == DummyRecord::keyForClass());

  edm::print_eventsetup_record_dependencies<DepRecord>(std::cout);
}

void testdependentrecord::dependentFinder1Test() {
  auto dummyProvider = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  const edm::EventID eID_1(1, 1, 1);
  const edm::IOVSyncValue sync_1(eID_1);
  const edm::EventID eID_3(1, 1, 3);
  const edm::ValidityInterval definedInterval(sync_1, edm::IOVSyncValue(eID_3));
  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(definedInterval);
  dummyProvider->addFinder(dummyFinder);

  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);
  finder.addProviderWeAreDependentOn(dummyProvider);

  CPPUNIT_ASSERT(definedInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2))));

  dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() ==
                 finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));

  const edm::EventID eID_5(1, 1, 5);
  const edm::IOVSyncValue sync_5(eID_5);
  const edm::ValidityInterval unknownedEndInterval(sync_5, edm::IOVSyncValue::invalidIOVSyncValue());
  dummyFinder->setInterval(unknownedEndInterval);

  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(unknownedEndInterval ==
                 finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));
}

void testdependentrecord::dependentFinder2Test() {
  auto dummyProvider1 = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  const edm::EventID eID_1(1, 1, 1);
  const edm::IOVSyncValue sync_1(eID_1);
  const edm::ValidityInterval definedInterval1(sync_1, edm::IOVSyncValue(edm::EventID(1, 1, 5)));
  dummyProvider1->setValidityInterval_forTesting(definedInterval1);

  auto dummyProvider2 = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  const edm::EventID eID_2(1, 1, 2);
  const edm::IOVSyncValue sync_2(eID_2);
  const edm::ValidityInterval definedInterval2(sync_2, edm::IOVSyncValue(edm::EventID(1, 1, 6)));
  dummyProvider2->setValidityInterval_forTesting(definedInterval2);

  const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                              std::min(definedInterval1.last(), definedInterval2.last()));

  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();

  DependentRecordIntervalFinder finder(depRecordKey);
  finder.addProviderWeAreDependentOn(dummyProvider1);
  finder.addProviderWeAreDependentOn(dummyProvider2);

  CPPUNIT_ASSERT(overlapInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));
}

void testdependentrecord::testIncomparibleIOVAlgorithm() {
  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);

  //test case where we have two providers, one synching on time the other on run/lumi/event
  auto dummyProviderEventID = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  auto dummyProviderTime = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  finder.addProviderWeAreDependentOn(dummyProviderEventID);
  finder.addProviderWeAreDependentOn(dummyProviderTime);

  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 5)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(6)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    // The algorithm selects incomparable IOVs from the two providers based on
    // the least estimated time difference from the start of changed IOVs.
    // The difference based on lumi numbers is zero in this case (assumes
    // 23 second lumis). The difference in times is also zero because it is
    // calculating in seconds and the lower 32 bits of the time argument to
    // the Timestamp constructor are dropped by the algorithm. So when finding
    // the closest both time and eventID estimates are zero. On a tie, the algorithm
    // selects the time IOV.
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(3))));

    // With IOVs that are not comparable we just continually get back the same
    // result when none of the IOVs changes. Next 3 just show that.

    //should give back same interval
    dummyProviderEventID->initializeForNewSyncValue();
    dummyProviderTime->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(4))));

    //should give back same interval
    dummyProviderEventID->initializeForNewSyncValue();
    dummyProviderTime->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(3))));

    //should give back same interval
    dummyProviderEventID->initializeForNewSyncValue();
    dummyProviderTime->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2))));
  }
  {
    //Change only run/lumi/event based provider. The algorithm picks
    //the closest of the IOVs that changed, so now gets the EventID based
    //IOV.
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 10)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    // Don't need to call initializeForNewSyncValue() if setValidityInterval_forTesting
    // was called.
    dummyProviderTime->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 6), edm::Timestamp(5))));
  }
  {
    //Change only time based provider
    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    dummyProviderEventID->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
  }
  //Change both but make run/lumi/event 'closer' by having same lumi
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 2, 11)),
                                           edm::IOVSyncValue(edm::EventID(1, 3, 20)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1ULL << 32)),
                                        edm::IOVSyncValue(edm::Timestamp(5ULL << 32)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(
        expectedIOV ==
        finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 2, 12), edm::Timestamp(3ULL << 32))));
  }
  //Change both but make time 'closer'
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 3, 21)),
                                           edm::IOVSyncValue(edm::EventID(1, 10, 40)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7ULL << 32)),
                                        edm::IOVSyncValue(edm::Timestamp(10ULL << 32)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(
        expectedIOV ==
        finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 4, 30), edm::Timestamp(8ULL << 32))));
  }
  //Change both but make run/lumi/event 'closer'
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 11, 41)),
                                           edm::IOVSyncValue(edm::EventID(1, 20, 60)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(11ULL << 32)),
                                        edm::IOVSyncValue(edm::Timestamp(100ULL << 32)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(
        expectedIOV ==
        finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 12, 50), edm::Timestamp(70ULL << 32))));
  }

  //Change both and make it ambiguous because of different run #
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(2, 1, 0)),
                                           edm::IOVSyncValue(edm::EventID(6, 0, 0)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(200ULL << 32)),
                                        edm::IOVSyncValue(edm::Timestamp(500ULL << 32)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(
        expectedIOV ==
        finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(4, 12, 50), edm::Timestamp(400ULL << 32))));
  }
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 10)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
  }
}

void testdependentrecord::testInvalidIOVAlgorithm() {
  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);

  auto dummyProviderEventID = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  auto dummyProviderTime = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  finder.addProviderWeAreDependentOn(dummyProviderEventID);
  finder.addProviderWeAreDependentOn(dummyProviderTime);

  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 6)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 10)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(7)), edm::IOVSyncValue(edm::Timestamp(10)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7))));
  }

  //check with invalid intervals
  const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                      edm::IOVSyncValue::invalidIOVSyncValue());
  {
    dummyProviderEventID->setValidityInterval_forTesting(invalid);
    dummyProviderTime->initializeForNewSyncValue();
    const edm::ValidityInterval expectedIOV(edm::IOVSyncValue(edm::Timestamp(7)),
                                            edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 11), edm::Timestamp(8))));
  }
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 12)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 20)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);
    dummyProviderTime->setValidityInterval_forTesting(invalid);
    const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 13), edm::Timestamp(11))));
  }
  {
    dummyProviderEventID->setValidityInterval_forTesting(invalid);
    dummyProviderTime->setValidityInterval_forTesting(invalid);

    dummyProviderEventID->initializeForNewSyncValue();
    CPPUNIT_ASSERT(invalid ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 13), edm::Timestamp(11))));
  }
}

void testdependentrecord::testInvalidIOVFirstTime() {
  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);

  auto dummyProviderEventID = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  auto dummyProviderTime = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  finder.addProviderWeAreDependentOn(dummyProviderEventID);
  finder.addProviderWeAreDependentOn(dummyProviderTime);

  const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                      edm::IOVSyncValue::invalidIOVSyncValue());
  {
    //check for bug which only happens the first time we synchronize
    // have the second one invalid
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 6)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    dummyProviderTime->setValidityInterval_forTesting(invalid);

    const edm::ValidityInterval expectedIOV(iovEventID.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(1))));
  }
  {
    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(2)), edm::IOVSyncValue(edm::Timestamp(6)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    dummyProviderEventID->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(3))));
  }
}

void testdependentrecord::testInvalidIOVFirstEventID() {
  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);

  auto dummyProviderEventID = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  auto dummyProviderTime = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  finder.addProviderWeAreDependentOn(dummyProviderEventID);
  finder.addProviderWeAreDependentOn(dummyProviderTime);

  const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                      edm::IOVSyncValue::invalidIOVSyncValue());
  {
    //check for bug which only happens the first time we synchronize
    // have the  first one invalid
    dummyProviderEventID->setValidityInterval_forTesting(invalid);

    const edm::ValidityInterval iovTime(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(6)));
    dummyProviderTime->setValidityInterval_forTesting(iovTime);

    const edm::ValidityInterval expectedIOV(iovTime.first(), edm::IOVSyncValue::invalidIOVSyncValue());

    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(3))));
  }
  {
    const edm::ValidityInterval iovEventID(edm::IOVSyncValue(edm::EventID(1, 1, 5)),
                                           edm::IOVSyncValue(edm::EventID(1, 1, 10)));
    dummyProviderEventID->setValidityInterval_forTesting(iovEventID);

    const edm::ValidityInterval expectedIOV(edm::IOVSyncValue(edm::Timestamp(1)),
                                            edm::IOVSyncValue::invalidIOVSyncValue());
    dummyProviderTime->initializeForNewSyncValue();
    CPPUNIT_ASSERT(expectedIOV ==
                   finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(4))));
  }
}

void testdependentrecord::timeAndRunTest() {
  edm::ParameterSet pset = createDummyPset();
  {
    SynchronousEventSetupsController controller;
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
    provider.add(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 5))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepOn2RecordProxyProvider>();
    provider.add(depProv);

    std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
    dummy2Finder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue(edm::Timestamp(5))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));

    {
      edm::ESParentContext parentC;
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
      const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2)));
      const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id2 = eventSetup2.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id2);

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(2)));
      const edm::EventSetup eventSetup3(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id3 = eventSetup3.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id3);

      dummy2Finder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(6)), edm::IOVSyncValue(edm::Timestamp(10))));

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(7)));
      const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id4 = eventSetup4.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 != id4);

      dummyFinder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 6)), edm::IOVSyncValue(edm::EventID(1, 1, 10))));

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(8)));
      const edm::EventSetup eventSetup5(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id5 = eventSetup5.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id4 != id5);
    }
  }

  {
    //check that going all the way through EventSetup works properly
    // using two records with open ended IOVs
    SynchronousEventSetupsController controller;
    EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

    std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
    provider.add(dummyProv);

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue::invalidIOVSyncValue()));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepOn2RecordProxyProvider>();
    provider.add(depProv);

    std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
    dummy2Finder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(1)), edm::IOVSyncValue::invalidIOVSyncValue()));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));
    {
      edm::ESParentContext parentC;
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
      const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(2)));
      const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id2 = eventSetup2.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id2);

      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 2), edm::Timestamp(2)));
      const edm::EventSetup eventSetup3(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id3 = eventSetup3.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id3);

      dummy2Finder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::Timestamp(6)), edm::IOVSyncValue::invalidIOVSyncValue()));
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4), edm::Timestamp(7)));
      const edm::EventSetup eventSetup4(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id4 = eventSetup4.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 != id4);

      dummyFinder->setInterval(
          edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 6)), edm::IOVSyncValue::invalidIOVSyncValue()));
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(8)));
      const edm::EventSetup eventSetup5(provider.eventSetupImpl(), 0, nullptr, parentC);
      long long id5 = eventSetup5.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id4 != id5);
    }
  }
}

void testdependentrecord::dependentSetproviderTest() {
  auto depProvider = std::make_unique<EventSetupRecordProvider>(DepRecord::keyForClass(), &activityRegistry);

  auto dummyProvider = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(
      edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  dummyProvider->addFinder(dummyFinder);

  CPPUNIT_ASSERT(*(depProvider->dependentRecords().begin()) == dummyProvider->key());

  std::vector<std::shared_ptr<EventSetupRecordProvider>> providers;
  providers.push_back(dummyProvider);
  depProvider->setDependentProviders(providers);
}

void testdependentrecord::getTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
  provider.add(dummyProv);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(
      edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepRecordProxyProvider>();
  provider.add(depProv);
  edm::ESParentContext parentC;
  {
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
    const DepRecord& depRecord = eventSetup1.get<DepRecord>();

    depRecord.getRecord<DummyRecord>();

    auto dr = depRecord.tryToGetRecord<DummyRecord>();
    CPPUNIT_ASSERT(dr.has_value());
  }
  {
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4)));
    const edm::EventSetup eventSetup2(provider.eventSetupImpl(), 0, nullptr, parentC);
    CPPUNIT_ASSERT_THROW(eventSetup2.get<DepRecord>(), edm::eventsetup::NoRecordException<DepRecord>);
  }
}

namespace {
  template <typename RECORD>
  struct DummyDataConsumer : public edm::EDConsumerBase {
    explicit DummyDataConsumer(edm::ESInputTag const& iTag)
        : m_token{esConsumes<edm::eventsetup::test::DummyData, RECORD>(iTag)} {}

    void prefetch(edm::EventSetupImpl const& iImpl) const {
      auto const& recs = this->esGetTokenRecordIndicesVector(edm::Transition::Event);
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          edm::ServiceToken token;
          rec->prefetchAsync(
              edm::WaitingTaskHolder(group, &waitTask), proxies[i], &iImpl, token, edm::ESParentContext{});
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
      auto const& proxies = this->esGetTokenIndicesVector(edm::Transition::Event);
      for (size_t i = 0; i != proxies.size(); ++i) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          oneapi::tbb::task_group group;
          edm::FinalWaitingTask waitTask{group};
          edm::ServiceToken token;
          rec->prefetchAsync(
              edm::WaitingTaskHolder(group, &waitTask), proxies[i], &iImpl, token, edm::ESParentContext{});
          waitTask.wait();
        }
      }
    }

    edm::ESGetToken<edm::eventsetup::test::DummyData, RECORD1> m_token1;
    edm::ESGetToken<edm::eventsetup::test::DummyData, RECORD2> m_token2;
  };
}  // namespace

void testdependentrecord::getDataWithESGetTokenTest() {
  using edm::eventsetup::test::DummyData;
  using edm::eventsetup::test::DummyProxyProvider;
  DummyData kGood{1};
  DummyData kBad{0};

  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

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
      edm::eventsetup::ComponentDescription description("DepRecordProxyProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordProxyProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }

    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& data = eventSetup.getData(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();
      auto const& data = depRecord.get(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data = depRecord.get(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data.value_);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data = depRecord.get(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data.value_);
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      auto const& data1 = depRecord.get(consumer.m_token1);
      CPPUNIT_ASSERT(kBad.value_ == data1.value_);

      auto const& data2 = depRecord.get(consumer.m_token2);
      CPPUNIT_ASSERT(kGood.value_ == data2.value_);
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      CPPUNIT_ASSERT_THROW(depRecord.get(consumer.m_token), cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      CPPUNIT_ASSERT_THROW(depRecord.get(consumer.m_token), cms::Exception);
    }
  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testdependentrecord::getHandleWithESGetTokenTest() {
  using edm::eventsetup::test::DummyData;
  using edm::eventsetup::test::DummyProxyProvider;
  DummyData kGood{1};
  DummyData kBad{0};

  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

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
      edm::eventsetup::ComponentDescription description("DepRecordProxyProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordProxyProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));

    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESHandle<DummyData> data = eventSetup.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data = depRecord.getHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESHandle<DummyData> data1 = depRecord.getHandle(consumer.m_token1);
      CPPUNIT_ASSERT(kBad.value_ == data1->value_);
      const edm::eventsetup::ComponentDescription* desc = data1.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");

      edm::ESHandle<DummyData> data2 = depRecord.getHandle(consumer.m_token2);
      CPPUNIT_ASSERT(kGood.value_ == data2->value_);
      desc = data2.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      CPPUNIT_ASSERT(not depRecord.getHandle(consumer.m_token));
      CPPUNIT_ASSERT_THROW(*depRecord.getHandle(consumer.m_token), cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      CPPUNIT_ASSERT(not depRecord.getHandle(consumer.m_token));
      CPPUNIT_ASSERT_THROW(*depRecord.getHandle(consumer.m_token), cms::Exception);
    }
  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testdependentrecord::getTransientHandleWithESGetTokenTest() {
  using edm::eventsetup::test::DummyData;
  using edm::eventsetup::test::DummyProxyProvider;
  DummyData kGood{1};
  DummyData kBad{0};

  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

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
      edm::eventsetup::ComponentDescription description("DepRecordProxyProviderWithData", "testTwo", 1, true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel", "blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DepRecordProxyProviderWithData>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
    dummyFinder->setInterval(
        edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
    provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    edm::ESParentContext parentC;
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      edm::ESTransientHandle<DummyData> data = eventSetup.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kBad.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(kGood.value_ == data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer2<DummyRecord, DepRecord> consumer{edm::ESInputTag{"", ""}, edm::ESInputTag{"", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data1 = depRecord.getTransientHandle(consumer.m_token1);
      CPPUNIT_ASSERT(kBad.value_ == data1->value_);
      const edm::eventsetup::ComponentDescription* desc = data1.description();
      CPPUNIT_ASSERT(desc->label_ == "testOne");

      edm::ESTransientHandle<DummyData> data2 = depRecord.getTransientHandle(consumer.m_token2);
      CPPUNIT_ASSERT(kGood.value_ == data2->value_);
      desc = data2.description();
      CPPUNIT_ASSERT(desc->label_ == "testTwo");
    }
    {
      DummyDataConsumer<DummyRecord> consumer{edm::ESInputTag{"DoesNotExist", ""}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(not data);
      CPPUNIT_ASSERT_THROW(*data, cms::Exception);
    }
    {
      DummyDataConsumer<DepRecord> consumer{edm::ESInputTag{"DoesNotExist", "blah"}};
      consumer.updateLookup(provider.recordsToProxyIndices());
      consumer.prefetch(provider.eventSetupImpl());
      const edm::EventSetup eventSetup{provider.eventSetupImpl(),
                                       static_cast<unsigned int>(edm::Transition::Event),
                                       consumer.esGetTokenIndices(edm::Transition::Event),
                                       parentC};
      auto const& depRecord = eventSetup.get<DepRecord>();

      edm::ESTransientHandle<DummyData> data = depRecord.getTransientHandle(consumer.m_token);
      CPPUNIT_ASSERT(not data);
      CPPUNIT_ASSERT_THROW(*data, cms::Exception);
    }
  } catch (const cms::Exception& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}

void testdependentrecord::oneOfTwoRecordTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
  provider.add(dummyProv);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(
      edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepOn2RecordProxyProvider>();
  provider.add(depProv);
  {
    edm::ESParentContext parentC;
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
    const DepOn2Record& depRecord = eventSetup1.get<DepOn2Record>();

    depRecord.getRecord<DummyRecord>();
    CPPUNIT_ASSERT_THROW(depRecord.getRecord<Dummy2Record>(), edm::eventsetup::NoRecordException<Dummy2Record>);

    try {
      depRecord.getRecord<Dummy2Record>();
    } catch (edm::eventsetup::NoRecordException<Dummy2Record>& e) {
      //make sure that the record name appears in the error message.
      CPPUNIT_ASSERT(0 != strstr(e.what(), "DepOn2Record"));
      CPPUNIT_ASSERT(0 != strstr(e.what(), "Dummy2Record"));
      // std::cout<<e.what()<<std::endl;
    }
  }
}

void testdependentrecord::resetTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
  provider.add(dummyProv);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(
      edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));

  std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepRecordProxyProvider>();
  provider.add(depProv);
  {
    edm::ESParentContext parentC;
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
    const DepRecord& depRecord = eventSetup1.get<DepRecord>();
    unsigned long long depCacheID = depRecord.cacheIdentifier();
    const DummyRecord& dummyRecord = depRecord.getRecord<DummyRecord>();
    unsigned long long dummyCacheID = dummyRecord.cacheIdentifier();

    provider.resetRecordPlusDependentRecords(dummyRecord.key());
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    CPPUNIT_ASSERT(dummyCacheID != dummyRecord.cacheIdentifier());
    CPPUNIT_ASSERT(depCacheID != depRecord.cacheIdentifier());
  }
}

void testdependentrecord::alternateFinderTest() {
  auto dummyProvider = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  const edm::EventID eID_1(1, 1, 1);
  const edm::IOVSyncValue sync_1(eID_1);
  const edm::EventID eID_3(1, 1, 3);
  const edm::IOVSyncValue sync_3(eID_3);
  const edm::EventID eID_4(1, 1, 4);
  const edm::ValidityInterval definedInterval(sync_1, edm::IOVSyncValue(eID_4));
  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  dummyFinder->setInterval(definedInterval);
  dummyProvider->addFinder(dummyFinder);

  std::shared_ptr<DepRecordFinder> depFinder = std::make_shared<DepRecordFinder>();
  const edm::EventID eID_2(1, 1, 2);
  const edm::IOVSyncValue sync_2(eID_2);
  const edm::ValidityInterval depInterval(sync_1, sync_2);
  depFinder->setInterval(depInterval);

  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);
  finder.setAlternateFinder(depFinder);
  finder.addProviderWeAreDependentOn(dummyProvider);

  CPPUNIT_ASSERT(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1))));

  const edm::ValidityInterval dep2Interval(sync_3, edm::IOVSyncValue(eID_4));
  depFinder->setInterval(dep2Interval);

  /*const edm::ValidityInterval tempIOV = */
  finder.findIntervalFor(depRecordKey, sync_3);
  //std::cout <<  tempIOV.first().eventID()<<" to "<<tempIOV.last().eventID() <<std::endl;
  CPPUNIT_ASSERT(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3));

  dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
  depFinder->setInterval(edm::ValidityInterval::invalidInterval());

  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() ==
                 finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));

  const edm::EventID eID_6(1, 1, 6);
  const edm::IOVSyncValue sync_6(eID_6);
  const edm::ValidityInterval unknownedEndInterval(sync_6, edm::IOVSyncValue::invalidIOVSyncValue());
  dummyFinder->setInterval(unknownedEndInterval);

  const edm::EventID eID_7(1, 1, 7);
  const edm::IOVSyncValue sync_7(eID_7);
  const edm::ValidityInterval iov6_7(sync_6, sync_7);
  depFinder->setInterval(iov6_7);

  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(unknownedEndInterval == finder.findIntervalFor(depRecordKey, sync_6));

  //see if dependent record can override the finder
  dummyFinder->setInterval(depInterval);
  depFinder->setInterval(definedInterval);
  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1))));

  dummyFinder->setInterval(dep2Interval);
  dummyProvider->initializeForNewSyncValue();
  CPPUNIT_ASSERT(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3));
}

void testdependentrecord::invalidRecordTest() {
  auto dummyProvider1 = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);

  const edm::ValidityInterval invalid(edm::IOVSyncValue::invalidIOVSyncValue(),
                                      edm::IOVSyncValue::invalidIOVSyncValue());

  dummyProvider1->setValidityInterval_forTesting(invalid);

  auto dummyProvider2 = std::make_shared<EventSetupRecordProvider>(DummyRecord::keyForClass(), &activityRegistry);
  dummyProvider2->setValidityInterval_forTesting(invalid);

  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);
  finder.addProviderWeAreDependentOn(dummyProvider1);
  finder.addProviderWeAreDependentOn(dummyProvider2);

  CPPUNIT_ASSERT(invalid == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2))));

  const edm::EventID eID_1(1, 1, 5);
  const edm::IOVSyncValue sync_1(eID_1);
  const edm::ValidityInterval definedInterval1(sync_1, edm::IOVSyncValue(edm::EventID(1, 1, 10)));
  const edm::EventID eID_2(1, 1, 2);
  const edm::IOVSyncValue sync_2(eID_2);
  const edm::ValidityInterval definedInterval2(sync_2, edm::IOVSyncValue(edm::EventID(1, 1, 6)));
  dummyProvider2->setValidityInterval_forTesting(definedInterval2);

  const edm::ValidityInterval openEnded1(definedInterval2.first(), edm::IOVSyncValue::invalidIOVSyncValue());

  dummyProvider1->initializeForNewSyncValue();
  dummyProvider2->initializeForNewSyncValue();
  CPPUNIT_ASSERT(openEnded1 == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 4))));

  dummyProvider1->setValidityInterval_forTesting(definedInterval1);

  const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                              std::min(definedInterval1.last(), definedInterval2.last()));

  dummyProvider1->initializeForNewSyncValue();
  dummyProvider2->initializeForNewSyncValue();
  CPPUNIT_ASSERT(overlapInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));

  dummyProvider2->setValidityInterval_forTesting(invalid);
  const edm::ValidityInterval openEnded2(definedInterval1.first(), edm::IOVSyncValue::invalidIOVSyncValue());

  dummyProvider1->initializeForNewSyncValue();
  dummyProvider2->initializeForNewSyncValue();
  CPPUNIT_ASSERT(openEnded2 == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 7))));
}

void testdependentrecord::extendIOVTest() {
  SynchronousEventSetupsController controller;
  edm::ParameterSet pset = createDummyPset();
  EventSetupProvider& provider = *controller.makeProvider(pset, &activityRegistry);

  std::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv = std::make_shared<DummyProxyProvider>();
  provider.add(dummyProv);

  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();

  edm::IOVSyncValue startSyncValue{edm::EventID{1, 1, 1}};
  dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 5}}});
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>{dummyFinder});

  std::shared_ptr<edm::eventsetup::DataProxyProvider> depProv = std::make_shared<DepOn2RecordProxyProvider>();
  provider.add(depProv);

  edm::ESParentContext parentC;
  std::shared_ptr<Dummy2RecordFinder> dummy2Finder = std::make_shared<Dummy2RecordFinder>();
  dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 6}}});
  provider.add(std::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummy2Finder));
  {
    controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1), edm::Timestamp(1)));
    const edm::EventSetup eventSetup1(provider.eventSetupImpl(), 0, nullptr, parentC);
    unsigned long long id1 = eventSetup1.get<DepOn2Record>().cacheIdentifier();
    CPPUNIT_ASSERT(id1 == eventSetup1.get<DummyRecord>().cacheIdentifier());
    CPPUNIT_ASSERT(id1 == eventSetup1.get<Dummy2Record>().cacheIdentifier());

    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 5), edm::Timestamp(2)));
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id);
      CPPUNIT_ASSERT(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
    }
    //extend the IOV DummyRecord while Dummy2Record still covers this range
    dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 7}}});
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 6), edm::Timestamp(7)));
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id);
      CPPUNIT_ASSERT(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
    }

    //extend the IOV Dummy2Record while DummyRecord still covers this range
    dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 7}}});

    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 7), edm::Timestamp(7)));
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id);
      CPPUNIT_ASSERT(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
    }
    //extend the both IOVs
    dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 8}}});

    dummyFinder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 8}}});
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 8), edm::Timestamp(7)));
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 == id);
      CPPUNIT_ASSERT(id1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
    }
    //extend only one and create a new IOV for the other
    dummy2Finder->setInterval(edm::ValidityInterval{startSyncValue, edm::IOVSyncValue{edm::EventID{1, 1, 9}}});

    dummyFinder->setInterval(
        edm::ValidityInterval{edm::IOVSyncValue{edm::EventID{1, 1, 9}}, edm::IOVSyncValue{edm::EventID{1, 1, 9}}});
    {
      controller.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 9), edm::Timestamp(7)));
      const edm::EventSetup eventSetup(provider.eventSetupImpl(), 0, nullptr, parentC);
      unsigned long long id = eventSetup.get<DepOn2Record>().cacheIdentifier();
      CPPUNIT_ASSERT(id1 + 1 == id);
      CPPUNIT_ASSERT(id1 + 1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
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
      CPPUNIT_ASSERT(id1 + 2 == id);
      CPPUNIT_ASSERT(id1 + 1 == eventSetup.get<DummyRecord>().cacheIdentifier());
      CPPUNIT_ASSERT(id1 + 1 == eventSetup.get<Dummy2Record>().cacheIdentifier());
    }
  }
}
