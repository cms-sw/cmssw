// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     ConcurrentIOVESSource
//
// Implementation:
//     ESSource used for tests of Framework support for
//     concurrent IOVs in the EventSetup system
//
// Original Author:  W. David Dagenhart
//         Created:  21 March 2019

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/Integration/test/IOVTestInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/WallclockTimer.h"

#include <memory>
#include <vector>

namespace edmtest {

  class ConcurrentIOVESSource : public edm::EventSetupRecordIntervalFinder, public edm::ESProducer {
  public:
    ConcurrentIOVESSource(edm::ParameterSet const&);

    std::unique_ptr<IOVTestInfo> produce(ESTestRecordI const&);
    std::unique_ptr<ESTestDataA> produceA(ESTestRecordA const&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void setIntervalFor(edm::eventsetup::EventSetupRecordKey const&,
                        edm::IOVSyncValue const&,
                        edm::ValidityInterval&) override;

    bool isConcurrentFinder() const override { return concurrentFinder_; }

    // These are thread safe because after the constructor we do not
    // modify their state.
    const bool iovIsTime_;
    std::set<edm::IOVSyncValue> setOfIOV_;
    std::set<edm::IOVSyncValue> setOfInvalidIOV_;
    const bool concurrentFinder_;
    const bool testForceESSourceMode_;
    const bool findForRecordA_;
    edm::WallclockTimer wallclockTimer_;

    // Be careful with this. It is modified in setIntervalFor
    // and the setIntervalFor function is called serially (nonconcurrent
    // with itself). But it is not thread safe to use this data member
    // in the produce methods unless concurrentFinder_ is false.
    edm::ValidityInterval validityInterval_;
  };

  ConcurrentIOVESSource::ConcurrentIOVESSource(edm::ParameterSet const& pset)
      : iovIsTime_(!pset.getParameter<bool>("iovIsRunNotTime")),
        concurrentFinder_(pset.getParameter<bool>("concurrentFinder")),
        testForceESSourceMode_(pset.getParameter<bool>("testForceESSourceMode")),
        findForRecordA_(pset.getParameter<bool>("findForRecordA")) {
    wallclockTimer_.start();

    std::vector<unsigned int> temp(pset.getParameter<std::vector<unsigned int>>("firstValidLumis"));
    for (auto val : temp) {
      if (iovIsTime_) {
        setOfIOV_.insert(edm::IOVSyncValue(edm::Timestamp(val)));
      } else {
        setOfIOV_.insert(edm::IOVSyncValue(edm::EventID(1, val, 0)));
      }
    }

    std::vector<unsigned int> tempInvalid(pset.getParameter<std::vector<unsigned int>>("invalidLumis"));
    for (auto val : tempInvalid) {
      if (iovIsTime_) {
        setOfInvalidIOV_.insert(edm::IOVSyncValue(edm::Timestamp(val)));
      } else {
        setOfInvalidIOV_.insert(edm::IOVSyncValue(edm::EventID(1, val, 0)));
      }
    }
    this->findingRecord<ESTestRecordI>();
    setWhatProduced(this);
    if (findForRecordA_) {
      this->findingRecord<ESTestRecordA>();
      setWhatProduced(this, &ConcurrentIOVESSource::produceA);
    }
  }

  std::unique_ptr<IOVTestInfo> ConcurrentIOVESSource::produce(ESTestRecordI const& record) {
    auto data = std::make_unique<IOVTestInfo>();

    edm::ValidityInterval iov = record.validityInterval();
    edm::LogAbsolute("ConcurrentIOVESSource")
        << "ConcurrentIOVESSource::produce startIOV = " << iov.first().luminosityBlockNumber()
        << " endIOV = " << iov.last().luminosityBlockNumber() << " IOV index = " << record.iovIndex()
        << " cache identifier = " << record.cacheIdentifier() << " time = " << wallclockTimer_.realTime();

    if (!concurrentFinder_) {
      if (validityInterval_ != iov) {
        throw cms::Exception("TestError")
            << "ConcurrentIOVESSource::produce, testing as nonconcurrent finder and IOV changed!";
      }
    }

    data->iovStartLumi_ = iov.first().luminosityBlockNumber();
    data->iovEndLumi_ = iov.last().luminosityBlockNumber();
    data->iovIndex_ = record.iovIndex();
    data->cacheIdentifier_ = record.cacheIdentifier();
    return data;
  }

  std::unique_ptr<ESTestDataA> ConcurrentIOVESSource::produceA(ESTestRecordA const& record) {
    edm::ValidityInterval iov = record.validityInterval();
    if (!testForceESSourceMode_ && record.iovIndex() != 0) {
      // This criteria should never fail because the EventSetupRecord class
      // is hard coded to allow only one IOV at a time.
      throw cms::Exception("TestError")
          << "ConcurrentIOVESSource::produce, more than one concurrent IOV for type ESTestRecordA!";
    }
    edm::LogAbsolute("ConcurrentIOVESSource")
        << "ConcurrentIOVESSource::produceA startIOV = " << iov.first().luminosityBlockNumber()
        << " endIOV = " << iov.last().luminosityBlockNumber() << " IOV index = " << record.iovIndex()
        << " cache identifier = " << record.cacheIdentifier() << " time = " << wallclockTimer_.realTime();
    return std::make_unique<ESTestDataA>(0);
  }

  void ConcurrentIOVESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    std::vector<unsigned int> emptyVector;
    desc.add<bool>("iovIsRunNotTime", true);
    desc.add<bool>("concurrentFinder", true);
    desc.add<bool>("testForceESSourceMode", false);
    desc.add<bool>("findForRecordA", false);
    desc.add<std::vector<unsigned int>>("firstValidLumis", emptyVector);
    desc.add<std::vector<unsigned int>>("invalidLumis", emptyVector);
    descriptions.addDefault(desc);
  }

  void ConcurrentIOVESSource::setIntervalFor(edm::eventsetup::EventSetupRecordKey const& key,
                                             edm::IOVSyncValue const& syncValue,
                                             edm::ValidityInterval& interval) {
    interval = edm::ValidityInterval::invalidInterval();
    validityInterval_ = interval;

    for (auto const& invalidSyncValue : setOfInvalidIOV_) {
      if (syncValue == invalidSyncValue) {
        return;
      }
    }

    //if no intervals given, fail immediately
    if (setOfIOV_.empty()) {
      return;
    }

    std::pair<std::set<edm::IOVSyncValue>::iterator, std::set<edm::IOVSyncValue>::iterator> itFound =
        setOfIOV_.equal_range(syncValue);

    if (itFound.first == itFound.second) {
      if (itFound.first == setOfIOV_.begin()) {
        //request is before first valid interval, so fail
        return;
      }
      //go back one step
      --itFound.first;
    }
    edm::IOVSyncValue endOfInterval = edm::IOVSyncValue::endOfTime();

    if (itFound.second != setOfIOV_.end()) {
      if (iovIsTime_) {
        endOfInterval = edm::IOVSyncValue(edm::Timestamp(itFound.second->time().value() - 1));
      } else {
        endOfInterval = edm::IOVSyncValue(
            edm::EventID(1, itFound.second->eventID().luminosityBlock() - 1, edm::EventID::maxEventNumber()));
      }
    }
    interval = edm::ValidityInterval(*(itFound.first), endOfInterval);
    validityInterval_ = interval;
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_SOURCE(ConcurrentIOVESSource);
