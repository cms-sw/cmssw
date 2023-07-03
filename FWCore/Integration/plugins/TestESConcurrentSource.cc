// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     TestESConcurrentSource
//
// Implementation:
//     ESSource used for tests of Framework support for
//     ESSources and ESProducers. This is primarily focused
//     on the infrastructure used by CondDBESSource.
//
// Original Author:  C Jones
//         Created:  16 Dec 2021

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ESSourceProductResolverConcurrentBase.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/Integration/interface/IOVTestInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace edmtest {

  class TestESConcurrentSource;

  class TestESConcurrentSourceTestResolver : public edm::eventsetup::ESSourceProductResolverConcurrentBase {
  public:
    TestESConcurrentSourceTestResolver(TestESConcurrentSource* TestESConcurrentSource);

  private:
    void prefetch(edm::eventsetup::DataKey const&, edm::EventSetupRecordDetails) override;
    void initializeForNewIOV() override;
    void const* getAfterPrefetchImpl() const override;

    IOVTestInfo iovTestInfo_;
    TestESConcurrentSource* testESConcurrentSource_;
  };

  class TestESConcurrentSource : public edm::eventsetup::ESProductResolverProvider,
                                 public edm::EventSetupRecordIntervalFinder {
  public:
    using EventSetupRecordKey = edm::eventsetup::EventSetupRecordKey;
    explicit TestESConcurrentSource(edm::ParameterSet const&);
    ~TestESConcurrentSource() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void busyWait(char const* msg) const;

    void incrementCount() {
      auto const v = ++count_;
      auto m = maxCount_.load();
      while (m < v) {
        maxCount_.compare_exchange_strong(m, v);
      }
    }
    std::atomic<unsigned int> count_;
    std::atomic<unsigned int> maxCount_;
    std::atomic<unsigned int> count_setIntervalFor_;
    std::atomic<unsigned int> count_initializeForNewIOV_;

  private:
    bool isConcurrentFinder() const override { return true; }
    void setIntervalFor(EventSetupRecordKey const&, edm::IOVSyncValue const&, edm::ValidityInterval&) override;
    KeyedResolversVector registerResolvers(EventSetupRecordKey const&, unsigned int iovIndex) override;
    void initConcurrentIOVs(EventSetupRecordKey const&, unsigned int nConcurrentIOVs) override;

    std::set<edm::IOVSyncValue> setOfIOV_;
    const unsigned int iterations_;
    const double pi_;
    unsigned int expectedNumberOfConcurrentIOVs_;
    unsigned int nConcurrentIOVs_ = 0;
    bool checkIOVInitialization_;
  };

  TestESConcurrentSourceTestResolver::TestESConcurrentSourceTestResolver(TestESConcurrentSource* testESConcurrentSource)
      : edm::eventsetup::ESSourceProductResolverConcurrentBase(), testESConcurrentSource_(testESConcurrentSource) {}

  void TestESConcurrentSourceTestResolver::prefetch(edm::eventsetup::DataKey const& iKey,
                                                    edm::EventSetupRecordDetails iRecord) {
    testESConcurrentSource_->incrementCount();
    testESConcurrentSource_->busyWait((std::string("getImpl ") + iKey.name().value()).c_str());

    edm::ValidityInterval iov = iRecord.validityInterval();
    edm::LogAbsolute("TestESConcurrentSourceTestResolver")
        << "TestESConcurrentSourceTestResolver::getImpl '" << iKey.name().value()
        << "' startIOV = " << iov.first().luminosityBlockNumber() << " endIOV = " << iov.last().luminosityBlockNumber()
        << " IOV index = " << iRecord.iovIndex() << " cache identifier = " << iRecord.cacheIdentifier();

    iovTestInfo_.iovStartLumi_ = iov.first().luminosityBlockNumber();
    iovTestInfo_.iovEndLumi_ = iov.last().luminosityBlockNumber();
    iovTestInfo_.iovIndex_ = iRecord.iovIndex();
    iovTestInfo_.cacheIdentifier_ = iRecord.cacheIdentifier();

    --testESConcurrentSource_->count_;
  }

  void const* TestESConcurrentSourceTestResolver::getAfterPrefetchImpl() const { return &iovTestInfo_; }

  void TestESConcurrentSourceTestResolver::initializeForNewIOV() {
    edm::LogAbsolute("TestESConcurrentSourceTestResolver::initializeForNewIOV")
        << "TestESConcurrentSourceTestResolver::initializeForNewIOV";
    ++testESConcurrentSource_->count_initializeForNewIOV_;
  }

  TestESConcurrentSource::TestESConcurrentSource(edm::ParameterSet const& pset)
      : count_(0),
        maxCount_(0),
        count_setIntervalFor_(0),
        count_initializeForNewIOV_(0),
        iterations_(pset.getParameter<unsigned int>("iterations")),
        pi_(std::acos(-1)),
        expectedNumberOfConcurrentIOVs_(pset.getParameter<unsigned int>("expectedNumberOfConcurrentIOVs")),
        checkIOVInitialization_(pset.getParameter<bool>("checkIOVInitialization")) {
    std::vector<unsigned int> temp(pset.getParameter<std::vector<unsigned int>>("firstValidLumis"));
    for (auto val : temp) {
      setOfIOV_.insert(edm::IOVSyncValue(edm::EventID(1, val, 0)));
    }

    findingRecord<ESTestRecordI>();
    usingRecord<ESTestRecordI>();
  }

  TestESConcurrentSource::~TestESConcurrentSource() {
    edm::LogAbsolute("TestESConcurrentSource") << "max concurrency seen " << maxCount_.load();
  }

  void TestESConcurrentSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    std::vector<unsigned int> emptyVector;
    desc.add<unsigned int>("iterations", 10 * 1000 * 1000);
    desc.add<bool>("checkIOVInitialization", false);
    desc.add<unsigned int>("expectedNumberOfConcurrentIOVs", 0);
    desc.add<std::vector<unsigned int>>("firstValidLumis", emptyVector);
    descriptions.addDefault(desc);
  }

  void TestESConcurrentSource::setIntervalFor(EventSetupRecordKey const&,
                                              edm::IOVSyncValue const& syncValue,
                                              edm::ValidityInterval& iov) {
    if (checkIOVInitialization_) {
      // Note that this check should pass with the specific configuration where I enable
      // the check, but in general it does not have to be true. The counts are offset
      // by 1 because the beginRun IOV is invalid (no IOV initialization).
      if (count_setIntervalFor_ > 0 && count_initializeForNewIOV_ != 2 * (count_setIntervalFor_ - 1)) {
        throw cms::Exception("TestFailure") << "TestESConcurrentSource::setIntervalFor,"
                                            << " unexpected number of IOV initializations";
      }
    }
    incrementCount();
    ++count_setIntervalFor_;
    busyWait("setIntervalFor");
    iov = edm::ValidityInterval::invalidInterval();

    if (setOfIOV_.empty()) {
      --count_;
      return;
    }

    std::pair<std::set<edm::IOVSyncValue>::iterator, std::set<edm::IOVSyncValue>::iterator> itFound =
        setOfIOV_.equal_range(syncValue);

    if (itFound.first == itFound.second) {
      if (itFound.first == setOfIOV_.begin()) {
        //request is before first valid interval, so fail
        --count_;
        return;
      }
      //go back one step
      --itFound.first;
    }

    edm::IOVSyncValue endOfInterval = edm::IOVSyncValue::endOfTime();
    if (itFound.second != setOfIOV_.end()) {
      endOfInterval = edm::IOVSyncValue(
          edm::EventID(1, itFound.second->eventID().luminosityBlock() - 1, edm::EventID::maxEventNumber()));
    }
    iov = edm::ValidityInterval(*(itFound.first), endOfInterval);
    --count_;
  }

  edm::eventsetup::ESProductResolverProvider::KeyedResolversVector TestESConcurrentSource::registerResolvers(
      EventSetupRecordKey const&, unsigned int iovIndex) {
    if (expectedNumberOfConcurrentIOVs_ != 0 && nConcurrentIOVs_ != expectedNumberOfConcurrentIOVs_) {
      throw cms::Exception("TestFailure") << "TestESConcurrentSource::registerResolvers,"
                                          << " unexpected number of concurrent IOVs";
    }
    KeyedResolversVector keyedResolversVector;

    {
      edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<IOVTestInfo>(),
                                       edm::eventsetup::IdTags(""));
      keyedResolversVector.emplace_back(dataKey, std::make_shared<TestESConcurrentSourceTestResolver>(this));
    }
    {
      edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<IOVTestInfo>(),
                                       edm::eventsetup::IdTags("other"));
      keyedResolversVector.emplace_back(dataKey, std::make_shared<TestESConcurrentSourceTestResolver>(this));
    }

    return keyedResolversVector;
  }

  void TestESConcurrentSource::initConcurrentIOVs(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {
    edm::LogAbsolute("TestESConcurrentSource::initConcurrentIOVs")
        << "Start TestESConcurrentSource::initConcurrentIOVs " << nConcurrentIOVs << " " << key.name();
    if (EventSetupRecordKey::makeKey<ESTestRecordI>() != key) {
      throw cms::Exception("TestFailure") << "TestESConcurrentSource::initConcurrentIOVs,"
                                          << " unexpected EventSetupRecordKey";
    }
    if (expectedNumberOfConcurrentIOVs_ != 0 && nConcurrentIOVs != expectedNumberOfConcurrentIOVs_) {
      throw cms::Exception("TestFailure") << "TestESConcurrentSource::initConcurrentIOVs,"
                                          << " unexpected number of concurrent IOVs";
    }
    nConcurrentIOVs_ = nConcurrentIOVs;
  }

  void TestESConcurrentSource::busyWait(char const* msg) const {
    edm::LogAbsolute("TestESConcurrentSource::busyWait") << "Start TestESConcurrentSource::busyWait " << msg;
    double sum = 0.;
    const double stepSize = pi_ / iterations_;
    for (unsigned int i = 0; i < iterations_; ++i) {
      sum += stepSize * cos(i * stepSize);
    }
    edm::LogAbsolute("TestESConcurrentSource::busyWait")
        << "Stop TestESConcurrentSource::busyWait " << msg << " " << sum;
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_SOURCE(TestESConcurrentSource);
