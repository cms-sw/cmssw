// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     RunLumiESSource
//
// Implementation:
//     ESSource used for tests of Framework support for
//     the EventSetup system in run and lumi transitions
//
// Original Author:  W. David Dagenhart
//         Created:  18 April 2019

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "IOVTestInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

namespace edmtest {

  class RunLumiESSource : public edm::EventSetupRecordIntervalFinder, public edm::ESProducer {
  public:
    RunLumiESSource(edm::ParameterSet const&);

    std::unique_ptr<IOVTestInfo> produce(ESTestRecordC const&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void setIntervalFor(edm::eventsetup::EventSetupRecordKey const&,
                        edm::IOVSyncValue const&,
                        edm::ValidityInterval&) override;

    bool isConcurrentFinder() const override { return true; }
  };

  RunLumiESSource::RunLumiESSource(edm::ParameterSet const&) {
    findingRecord<ESTestRecordC>();
    setWhatProduced(this);
  }

  std::unique_ptr<IOVTestInfo> RunLumiESSource::produce(ESTestRecordC const& record) {
    auto data = std::make_unique<IOVTestInfo>();

    edm::ValidityInterval iov = record.validityInterval();
    edm::LogAbsolute("RunLumiESSource") << "RunLumiESSource::produce startIOV = " << iov.first().eventID().run() << ":"
                                        << iov.first().luminosityBlockNumber()
                                        << " endIOV = " << iov.last().eventID().run() << ":"
                                        << iov.last().luminosityBlockNumber() << " IOV index = " << record.iovIndex()
                                        << " cache identifier = " << record.cacheIdentifier();
    data->iovStartRun_ = iov.first().eventID().run();
    data->iovStartLumi_ = iov.first().luminosityBlockNumber();
    data->iovEndRun_ = iov.last().eventID().run();
    data->iovEndLumi_ = iov.last().luminosityBlockNumber();
    data->iovIndex_ = record.iovIndex();
    data->cacheIdentifier_ = record.cacheIdentifier();
    return data;
  }

  void RunLumiESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }

  void RunLumiESSource::setIntervalFor(edm::eventsetup::EventSetupRecordKey const&,
                                       edm::IOVSyncValue const& syncValue,
                                       edm::ValidityInterval& interval) {
    interval = edm::ValidityInterval(syncValue, syncValue);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_SOURCE(RunLumiESSource);
