// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     TestESSource
//
// Implementation:
//     ESSource used for tests of Framework support for
//     ESSources and ESProducers. This is primarily focused
//     on the infrastructure used by CondDBESSource.
//
// Original Author:  W. David Dagenhart
//         Created:  15 August 2019

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ESSourceDataProxyNonConcurrentBase.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "IOVTestInfo.h"
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
#include <mutex>

namespace edmtest {

  class TestESSource;

  class TestESSourceTestProxy : public edm::eventsetup::ESSourceDataProxyNonConcurrentBase {
  public:
    TestESSourceTestProxy(TestESSource* testESSource);

  private:
    void prefetch(edm::eventsetup::DataKey const&, edm::EventSetupRecordDetails) override;
    void initializeForNewIOV() override;
    void const* getAfterPrefetchImpl() const override;

    IOVTestInfo iovTestInfo_;
    TestESSource* testESSource_;
  };

  class TestESSource : public edm::eventsetup::DataProxyProvider, public edm::EventSetupRecordIntervalFinder {
  public:
    using EventSetupRecordKey = edm::eventsetup::EventSetupRecordKey;
    explicit TestESSource(edm::ParameterSet const&);
    ~TestESSource() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void busyWait(char const* msg) const;

    std::atomic<unsigned int> count_;
    std::atomic<unsigned int> count1_;
    std::atomic<unsigned int> count2_;
    edm::SerialTaskQueue queue_;
    std::mutex mutex_;

  private:
    bool isConcurrentFinder() const override { return true; }
    void setIntervalFor(EventSetupRecordKey const&, edm::IOVSyncValue const&, edm::ValidityInterval&) override;
    KeyedProxiesVector registerProxies(EventSetupRecordKey const&, unsigned int iovIndex) override;
    void initConcurrentIOVs(EventSetupRecordKey const&, unsigned int nConcurrentIOVs) override;

    std::set<edm::IOVSyncValue> setOfIOV_;
    const unsigned int iterations_;
    const double pi_;
    unsigned int expectedNumberOfConcurrentIOVs_;
    unsigned int nConcurrentIOVs_ = 0;
    bool checkIOVInitialization_;
  };

  TestESSourceTestProxy::TestESSourceTestProxy(TestESSource* testESSource)
      : edm::eventsetup::ESSourceDataProxyNonConcurrentBase(&testESSource->queue_, &testESSource->mutex_),
        testESSource_(testESSource) {}

  void TestESSourceTestProxy::prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails iRecord) {
    ++testESSource_->count_;
    if (testESSource_->count_.load() > 1) {
      throw cms::Exception("TestFailure") << "TestESSourceTestProxy::getImpl,"
                                          << " functions in mutex should not run concurrently";
    }
    testESSource_->busyWait("getImpl");

    edm::ValidityInterval iov = iRecord.validityInterval();
    edm::LogAbsolute("TestESSourceTestProxy")
        << "TestESSoureTestProxy::getImpl startIOV = " << iov.first().luminosityBlockNumber()
        << " endIOV = " << iov.last().luminosityBlockNumber() << " IOV index = " << iRecord.iovIndex()
        << " cache identifier = " << iRecord.cacheIdentifier();

    iovTestInfo_.iovStartLumi_ = iov.first().luminosityBlockNumber();
    iovTestInfo_.iovEndLumi_ = iov.last().luminosityBlockNumber();
    iovTestInfo_.iovIndex_ = iRecord.iovIndex();
    iovTestInfo_.cacheIdentifier_ = iRecord.cacheIdentifier();

    --testESSource_->count_;
  }

  void const* TestESSourceTestProxy::getAfterPrefetchImpl() const { return &iovTestInfo_; }

  void TestESSourceTestProxy::initializeForNewIOV() {
    edm::LogAbsolute("TestESSourceTestProxy::initializeForNewIOV") << "TestESSourceTestProxy::initializeForNewIOV";
    ++testESSource_->count2_;
  }

  TestESSource::TestESSource(edm::ParameterSet const& pset)
      : count_(0),
        count1_(0),
        count2_(0),
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

  TestESSource::~TestESSource() {}

  void TestESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    std::vector<unsigned int> emptyVector;
    desc.add<unsigned int>("iterations", 10 * 1000 * 1000);
    desc.add<bool>("checkIOVInitialization", false);
    desc.add<unsigned int>("expectedNumberOfConcurrentIOVs", 0);
    desc.add<std::vector<unsigned int>>("firstValidLumis", emptyVector);
    descriptions.addDefault(desc);
  }

  void TestESSource::setIntervalFor(EventSetupRecordKey const&,
                                    edm::IOVSyncValue const& syncValue,
                                    edm::ValidityInterval& iov) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (checkIOVInitialization_) {
      // Note that this check should pass with the specific configuration where I enable
      // the check, but in general it does not have to be true. The counts are offset
      // by 1 because the beginRun IOV is invalid (no IOV initialization).
      if (count1_ > 0 && count2_ + 1 != count1_) {
        throw cms::Exception("TestFailure") << "TestESSource::setIntervalFor,"
                                            << " unexpected number of IOV initializations";
      }
    }
    ++count_;
    ++count1_;
    if (count_.load() > 1) {
      throw cms::Exception("TestFailure") << "TestESSource::setIntervalFor,"
                                          << " functions in mutex should not run concurrently";
    }
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

  edm::eventsetup::DataProxyProvider::KeyedProxiesVector TestESSource::registerProxies(EventSetupRecordKey const&,
                                                                                       unsigned int iovIndex) {
    if (expectedNumberOfConcurrentIOVs_ != 0 && nConcurrentIOVs_ != expectedNumberOfConcurrentIOVs_) {
      throw cms::Exception("TestFailure") << "TestESSource::registerProxies,"
                                          << " unexpected number of concurrent IOVs";
    }
    KeyedProxiesVector keyedProxiesVector;

    edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<IOVTestInfo>(), edm::eventsetup::IdTags(""));
    keyedProxiesVector.emplace_back(dataKey, std::make_shared<TestESSourceTestProxy>(this));

    return keyedProxiesVector;
  }

  void TestESSource::initConcurrentIOVs(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {
    edm::LogAbsolute("TestESSource::initConcurrentIOVs")
        << "Start TestESSource::initConcurrentIOVs " << nConcurrentIOVs << " " << key.name();
    if (EventSetupRecordKey::makeKey<ESTestRecordI>() != key) {
      throw cms::Exception("TestFailure") << "TestESSource::initConcurrentIOVs,"
                                          << " unexpected EventSetupRecordKey";
    }
    if (expectedNumberOfConcurrentIOVs_ != 0 && nConcurrentIOVs != expectedNumberOfConcurrentIOVs_) {
      throw cms::Exception("TestFailure") << "TestESSource::initConcurrentIOVs,"
                                          << " unexpected number of concurrent IOVs";
    }
    nConcurrentIOVs_ = nConcurrentIOVs;
  }

  void TestESSource::busyWait(char const* msg) const {
    edm::LogAbsolute("TestESSource::busyWait") << "Start TestESSource::busyWait " << msg;
    double sum = 0.;
    const double stepSize = pi_ / iterations_;
    for (unsigned int i = 0; i < iterations_; ++i) {
      sum += stepSize * cos(i * stepSize);
    }
    edm::LogAbsolute("TestESSource::busyWait") << "Stop TestESSource::busyWait " << msg << " " << sum;
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_SOURCE(TestESSource);
