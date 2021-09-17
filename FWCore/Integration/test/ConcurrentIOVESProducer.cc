// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      ConcurrentIOVESProducer
//
/**\class edmtest::ConcurrentIOVESProducer

  Description: Used in tests of the concurrent IOV feature of the
  EventSetup system.
*/
// Original Author:  W. David Dagenhart
//         Created:  22 March 2019

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/Integration/test/IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

namespace edmtest {

  class ConcurrentIOVESProducer : public edm::ESProducer {
  public:
    ConcurrentIOVESProducer(edm::ParameterSet const&);

    std::unique_ptr<IOVTestInfo> produce(ESTestRecordI const&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::ESGetToken<IOVTestInfo, ESTestRecordI> token_;
  };

  ConcurrentIOVESProducer::ConcurrentIOVESProducer(edm::ParameterSet const&) {
    //auto collector = setWhatProduced(this);
    auto collector = setWhatProduced(this, "fromESProducer");
    token_ = collector.consumes<IOVTestInfo>(edm::ESInputTag{"", ""});
  }

  std::unique_ptr<IOVTestInfo> ConcurrentIOVESProducer::produce(ESTestRecordI const& record) {
    edm::ESHandle<IOVTestInfo> iovTestInfo = record.getHandle(token_);

    edm::ValidityInterval iov = record.validityInterval();
    if (iovTestInfo->iovStartLumi_ != iov.first().luminosityBlockNumber() ||
        iovTestInfo->iovEndLumi_ != iov.last().luminosityBlockNumber() || iovTestInfo->iovIndex_ != record.iovIndex() ||
        iovTestInfo->cacheIdentifier_ != record.cacheIdentifier()) {
      throw cms::Exception("TestFailure") << "ConcurrentIOVESProducer::ConcurrentIOVESProducer"
                                          << "read values do not agree with record";
    }

    auto data = std::make_unique<IOVTestInfo>();
    *data = *iovTestInfo;
    return data;
  }

  void ConcurrentIOVESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("concurrentIOVESProducer", desc);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(ConcurrentIOVESProducer);
