#ifndef Framework_DummyEventSetupRecordRetriever_h
#define Framework_DummyEventSetupRecordRetriever_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyEventSetupRecordRetriever
//
/**\class DummyEventSetupRecordRetriever DummyEventSetupRecordRetriever.h FWCore/Framework/interface/DummyEventSetupRecordRetriever.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr 22 14:14:09 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "FWCore/Framework/test/DummyEventSetupRecord.h"
#include "FWCore/Framework/test/DummyEventSetupData.h"

// forward declarations
namespace edm {
  class DummyEventSetupRecordRetriever : public EventSetupRecordIntervalFinder, public ESProducer {
  public:
    explicit DummyEventSetupRecordRetriever(edm::IOVSyncValue const& iStart = edm::IOVSyncValue::invalidIOVSyncValue())
        : m_start(iStart) {
      this->findingRecord<DummyEventSetupRecord>();
      setWhatProduced(this);
    }

    std::unique_ptr<DummyEventSetupData> produce(const DummyEventSetupRecord&) {
      return std::make_unique<DummyEventSetupData>(1);
    }

  protected:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                        const edm::IOVSyncValue& iTime,
                        edm::ValidityInterval& iInterval) final {
      if (m_start == edm::IOVSyncValue::invalidIOVSyncValue()) {
        iInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
      } else {
        edm::ValidityInterval temp(m_start, edm::IOVSyncValue::endOfTime());
        if (temp.validFor(iTime)) {
          iInterval = temp;
        } else {
          iInterval = edm::ValidityInterval();
        }
      }
    }

    bool isConcurrentFinder() const final { return true; }

  private:
    DummyEventSetupRecordRetriever(const DummyEventSetupRecordRetriever&) = delete;  // stop default

    const DummyEventSetupRecordRetriever& operator=(const DummyEventSetupRecordRetriever&) = delete;  // stop default

    // ---------- member data --------------------------------
    edm::IOVSyncValue m_start;
  };
}  // namespace edm
#endif
