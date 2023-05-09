// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      AcquireIntESProducer
//
/**\class edmtest::AcquireIntESProducer

  Description: Used in tests of the asynchronous ESProducer.
*/
// Original Author:  W. David Dagenhart
//         Created:  12 January 2023

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducerExternalWork.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/Integration/interface/IOVTestInfo.h"
#include "WaitingServer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <optional>
#include <unistd.h>
#include <vector>

namespace {
  constexpr int kAcquireTestValue = 11;
  constexpr int kAcquireTestValueUniquePtr1 = 101;
  constexpr int kAcquireTestValueUniquePtr2 = 102;
  constexpr int kAcquireTestValueOptional1 = 201;
  constexpr int kAcquireTestValueOptional2 = 202;
}  // namespace

namespace edmtest {

  class AcquireIntESProducer : public edm::ESProducerExternalWork {
  public:
    AcquireIntESProducer(edm::ParameterSet const&);

    ~AcquireIntESProducer() override;
    AcquireIntESProducer(const AcquireIntESProducer&) = delete;
    AcquireIntESProducer& operator=(const AcquireIntESProducer&) = delete;
    AcquireIntESProducer(AcquireIntESProducer&&) = delete;
    AcquireIntESProducer& operator=(AcquireIntESProducer&&) = delete;

    void initConcurrentIOVs(EventSetupRecordKey const&, unsigned int nConcurrentIOVs) override;

    int acquire(ESTestRecordI const&, edm::WaitingTaskWithArenaHolder);

    std::unique_ptr<ESTestDataI> produce(ESTestRecordI const&, int);

    std::unique_ptr<ESTestDataB> produceESTestDataB(ESTestRecordB const&);

    class TestValue {
    public:
      TestValue(int value) : value_(value) {}
      int value_;
    };

    std::unique_ptr<TestValue> acquireUniquePtr(ESTestRecordI const&, edm::WaitingTaskWithArenaHolder);

    std::unique_ptr<ESTestDataI> produceUniquePtr(ESTestRecordI const&, std::unique_ptr<TestValue>);

    std::optional<std::vector<TestValue>> acquireOptional(ESTestRecordI const&, edm::WaitingTaskWithArenaHolder);

    std::unique_ptr<ESTestDataI> produceOptional(ESTestRecordI const&, std::optional<std::vector<TestValue>>);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::ESGetToken<IOVTestInfo, ESTestRecordI> token_;
    std::vector<test_acquire::Cache> caches_;
    std::unique_ptr<test_acquire::WaitingServer> server_;
    const unsigned int numberOfIOVsToAccumulate_;
    const unsigned int secondsToWaitForWork_;
    std::vector<TestValue*> uniqueTestPointers_;
    std::vector<TestValue*> optionalTestPointers_;
  };

  AcquireIntESProducer::AcquireIntESProducer(edm::ParameterSet const& pset)
      : numberOfIOVsToAccumulate_(pset.getUntrackedParameter<unsigned int>("numberOfIOVsToAccumulate")),
        secondsToWaitForWork_(pset.getUntrackedParameter<unsigned int>("secondsToWaitForWork")) {
    auto collector = setWhatAcquiredProduced(this, "fromAcquireIntESProducer");
    token_ = collector.consumes<IOVTestInfo>(edm::ESInputTag{"", ""});

    setWhatProduced(this, &edmtest::AcquireIntESProducer::produceESTestDataB);

    setWhatAcquiredProduced(this,
                            &AcquireIntESProducer::acquireUniquePtr,
                            &AcquireIntESProducer::produceUniquePtr,
                            edm::es::Label("uniquePtr"));

    setWhatAcquiredProduced(this,
                            &AcquireIntESProducer::acquireOptional,
                            &AcquireIntESProducer::produceOptional,
                            edm::es::Label("optional"));
  }

  AcquireIntESProducer::~AcquireIntESProducer() {
    if (server_) {
      server_->stop();
    }
    server_.reset();
  }

  void AcquireIntESProducer::initConcurrentIOVs(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {
    if (key == EventSetupRecordKey::makeKey<ESTestRecordI>()) {
      caches_.resize(nConcurrentIOVs);
      server_ = std::make_unique<test_acquire::WaitingServer>(
          nConcurrentIOVs, numberOfIOVsToAccumulate_, secondsToWaitForWork_);
      server_->start();
      uniqueTestPointers_.resize(nConcurrentIOVs);
      optionalTestPointers_.resize(nConcurrentIOVs);
    }
  }

  int AcquireIntESProducer::acquire(ESTestRecordI const& record, edm::WaitingTaskWithArenaHolder holder) {
    usleep(200000);

    test_acquire::Cache& iovCache = caches_[record.iovIndex()];
    iovCache.retrieved().clear();
    iovCache.processed().clear();

    // Get some data and push it into the input cache for the ExternalWork.
    // There is no significance to the particular data we are using.
    // Using anything from the EventSetup would be good enough for the test.
    // I already had test modules around that would make IOVTestInfo
    // data, so that was easy to use. We put in known values and later
    // check that we get the expected result (they get incremented by one
    // to simulate some "external work", then summed in the produce method
    // calculate a result we can check easily).
    IOVTestInfo const& iovTestInfo = record.get(token_);
    std::vector<int>& retrieved = iovCache.retrieved();
    retrieved.push_back(iovTestInfo.iovStartRun_);
    retrieved.push_back(iovTestInfo.iovStartLumi_);
    retrieved.push_back(iovTestInfo.iovEndRun_);
    retrieved.push_back(iovTestInfo.iovEndLumi_);
    retrieved.push_back(iovTestInfo.cacheIdentifier_);

    server_->requestValuesAsync(record.iovIndex(), &iovCache.retrieved(), &iovCache.processed(), holder);

    edm::ValidityInterval iov = record.validityInterval();
    if (iovTestInfo.iovStartLumi_ != iov.first().luminosityBlockNumber() ||
        iovTestInfo.iovEndLumi_ != iov.last().luminosityBlockNumber() || iovTestInfo.iovIndex_ != record.iovIndex() ||
        iovTestInfo.cacheIdentifier_ != record.cacheIdentifier()) {
      throw cms::Exception("TestFailure") << "AcquireIntESProducer::acquire"
                                          << "read values do not agree with record";
    }
    return kAcquireTestValue;
  }

  std::unique_ptr<ESTestDataI> AcquireIntESProducer::produce(ESTestRecordI const& record, int valueReturnedByAcquire) {
    usleep(200000);

    if (valueReturnedByAcquire != kAcquireTestValue) {
      throw cms::Exception("TestFailure") << "AcquireIntESProducer::produce"
                                          << " unexpected value passed in as argument";
    }

    edm::ESHandle<IOVTestInfo> iovTestInfo = record.getHandle(token_);
    edm::ValidityInterval iov = record.validityInterval();
    if (iovTestInfo->iovStartLumi_ != iov.first().luminosityBlockNumber() ||
        iovTestInfo->iovEndLumi_ != iov.last().luminosityBlockNumber() || iovTestInfo->iovIndex_ != record.iovIndex() ||
        iovTestInfo->cacheIdentifier_ != record.cacheIdentifier()) {
      throw cms::Exception("TestFailure") << "AcquireIntESProducer::produce"
                                          << "read values do not agree with record";
    }

    test_acquire::Cache& iovCache = caches_[record.iovIndex()];
    int sum = 0;
    for (auto v : iovCache.processed()) {
      sum += v;
    }
    return std::make_unique<ESTestDataI>(sum);
  }

  std::unique_ptr<ESTestDataB> AcquireIntESProducer::produceESTestDataB(ESTestRecordB const&) {
    return std::make_unique<ESTestDataB>(11);
  }

  std::unique_ptr<AcquireIntESProducer::TestValue> AcquireIntESProducer::acquireUniquePtr(
      ESTestRecordI const& record, edm::WaitingTaskWithArenaHolder holder) {
    usleep(200000);
    auto returnValue = std::make_unique<TestValue>(kAcquireTestValueUniquePtr1);
    uniqueTestPointers_[record.iovIndex()] = returnValue.get();
    return returnValue;
  }

  std::unique_ptr<ESTestDataI> AcquireIntESProducer::produceUniquePtr(ESTestRecordI const& record,
                                                                      std::unique_ptr<TestValue> testValue) {
    usleep(200000);
    if (testValue.get() != uniqueTestPointers_[record.iovIndex()]) {
      throw cms::Exception("TestFailure") << "AcquireIntESProducer::produceUniquePtr"
                                          << " unexpected value passed in as argument";
    }
    return std::make_unique<ESTestDataI>(kAcquireTestValueUniquePtr2);
  }

  std::optional<std::vector<AcquireIntESProducer::TestValue>> AcquireIntESProducer::acquireOptional(
      ESTestRecordI const& record, edm::WaitingTaskWithArenaHolder holder) {
    usleep(200000);
    std::vector<TestValue> testVector;
    testVector.push_back(kAcquireTestValueOptional1);
    auto returnValue = std::make_optional<std::vector<TestValue>>(std::move(testVector));
    optionalTestPointers_[record.iovIndex()] = &returnValue.value()[0];
    return returnValue;
  }

  std::unique_ptr<ESTestDataI> AcquireIntESProducer::produceOptional(ESTestRecordI const& record,
                                                                     std::optional<std::vector<TestValue>> testValue) {
    usleep(200000);
    if (&testValue.value()[0] != optionalTestPointers_[record.iovIndex()]) {
      throw cms::Exception("TestFailure") << "AcquireIntESProducer::produceOptional"
                                          << " unexpected value passed in as argument";
    }
    return std::make_unique<ESTestDataI>(kAcquireTestValueOptional2);
  }

  void AcquireIntESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<unsigned int>("numberOfIOVsToAccumulate", 8);
    desc.addUntracked<unsigned int>("secondsToWaitForWork", 1);
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(AcquireIntESProducer);
